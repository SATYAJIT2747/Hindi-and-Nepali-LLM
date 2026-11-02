# =============================================================================
# V7 FINAL EVALUATION — EXACT V7 ARCHITECTURE (NEPALI)
# =============================================================================

import os
import sys
import json
from collections import defaultdict

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import torch
import torch.nn as nn
import torch.nn.functional as F
import sentencepiece as spm
from tqdm import tqdm


# =============================================================================
# CONFIG
# =============================================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def resolve_path(env_key, default_name):
    env_val = os.getenv(env_key)
    local_p = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data", default_name))
    if os.name == "nt":
        if env_val and os.path.exists(env_val):
            return env_val
        return local_p
    return env_val or local_p

CHECKPOINT_PATH = resolve_path("CHECKPOINT_PATH", "hard_best.pt")
TOKENIZER_PATH = resolve_path("TOKENIZER_PATH", "nepali_bpe_32000.model")
TEST_PATH = resolve_path("TEST_PATH", "hard_test.jsonl")


MAX_SEQ_LEN = 512
D_MODEL = 448
NHEAD = 8
DIM_FEEDFORWARD = 1024
NUM_LAYERS = 6
DROPOUT = 0.1


print("=" * 80)
print("V7 FINAL EVALUATION (NEPALI)")
print("=" * 80)
print(f"Device     : {DEVICE}")
print(f"Checkpoint : {CHECKPOINT_PATH}")
print(f"Test file  : {TEST_PATH}")
print()


if not os.path.exists(TOKENIZER_PATH):
    raise FileNotFoundError(f"\nTokenizer not found:\n{TOKENIZER_PATH}")

if not os.path.exists(TEST_PATH):
    raise FileNotFoundError(f"\nTest file not found:\n{TEST_PATH}")


# =============================================================================
# TOKENIZER
# =============================================================================

sp = spm.SentencePieceProcessor()
sp.load(TOKENIZER_PATH)

VOCAB_SIZE = sp.get_piece_size()
EOS_ID = sp.eos_id()
PAD_ID = sp.pad_id()

print("Tokenizer loaded.")
print(f"Vocabulary size: {VOCAB_SIZE}")
print(f"EOS ID: {EOS_ID}")
print(f"PAD ID: {PAD_ID}")
print()


# =============================================================================
# MODEL ARCHITECTURE
# =============================================================================

class CausalSelfAttention(nn.Module):

    def __init__(self, d_model, nhead, max_seq_len):
        super().__init__()
        assert d_model % nhead == 0

        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        mask = torch.triu(
            torch.ones(max_seq_len, max_seq_len),
            diagonal=1
        ).bool()

        self.register_buffer(
            "causal_mask",
            mask.view(1, 1, max_seq_len, max_seq_len),
            persistent=True
        )

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.nhead, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.nhead, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.nhead, self.head_dim).transpose(1, 2)

        attn_mask = self.causal_mask[:, :, :T, :T]

        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=~attn_mask,
            dropout_p=0.0,
            is_causal=False
        )

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(y)


class TransformerBlock(nn.Module):

    def __init__(
        self,
        d_model=448,
        nhead=8,
        dim_feedforward=1024,
        max_seq_len=512,
        dropout=0.1
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, nhead, max_seq_len)
        self.ln2 = nn.LayerNorm(d_model)

        self.ffn = nn.ModuleDict({
            "fc1": nn.Linear(d_model, dim_feedforward),
            "fc2": nn.Linear(dim_feedforward, d_model)
        })
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.dropout(self.attn(self.ln1(x)))
        x = x + self.dropout(
            self.ffn["fc2"](
                F.gelu(self.ffn["fc1"](self.ln2(x)))
            )
        )
        return x


class NepaliGPT(nn.Module):

    def __init__(self):
        super().__init__()
        self.vocab_size = VOCAB_SIZE
        self.d_model = D_MODEL
        self.max_seq_len = MAX_SEQ_LEN

        self.tok_emb = nn.Embedding(self.vocab_size, self.d_model)
        self.pos_emb = nn.Embedding(self.max_seq_len, self.d_model)
        self.drop = nn.Dropout(DROPOUT)

        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=D_MODEL,
                nhead=NHEAD,
                dim_feedforward=DIM_FEEDFORWARD,
                max_seq_len=MAX_SEQ_LEN,
                dropout=DROPOUT
            )
            for _ in range(NUM_LAYERS)
        ])

        self.ln_f = nn.LayerNorm(self.d_model)
        self.lm_head = nn.Linear(self.d_model, self.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight

    def forward(self, idx):
        B, T = idx.shape
        if T > self.max_seq_len:
            raise ValueError(f"Sequence length {T} exceeds max_seq_len {self.max_seq_len}")

        pos = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        x = self.drop(x)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        return self.lm_head(x)


HindiGPT = NepaliGPT


# =============================================================================
# PREDICTION HELPERS
# =============================================================================

@torch.no_grad()
def predict_first_token(model, prompt_text):
    prompt_ids = sp.encode_as_ids(prompt_text)
    if len(prompt_ids) == 0:
        raise ValueError("Prompt produced no tokens.")
    if len(prompt_ids) > MAX_SEQ_LEN:
        prompt_ids = prompt_ids[-MAX_SEQ_LEN:]

    x = torch.tensor([prompt_ids], dtype=torch.long, device=DEVICE)
    logits = model(x)
    return logits[0, -1].argmax().item()


@torch.no_grad()
def generate_answer(model, prompt_text, max_new_tokens=12):
    prompt_ids = sp.encode_as_ids(prompt_text)
    if len(prompt_ids) == 0:
        raise ValueError("Prompt produced no tokens.")

    generated = list(prompt_ids)
    for _ in range(max_new_tokens):
        context = generated[-MAX_SEQ_LEN:]
        x = torch.tensor([context], dtype=torch.long, device=DEVICE)
        logits = model(x)
        next_id = logits[0, -1].argmax().item()
        generated.append(next_id)
        if next_id == EOS_ID:
            break

    answer_ids = generated[len(prompt_ids):]
    if EOS_ID in answer_ids:
        answer_ids = answer_ids[:answer_ids.index(EOS_ID)]

    return sp.decode_ids(answer_ids).strip()


# =============================================================================
# EVALUATION RUNNER
# =============================================================================

def load_and_eval():
    print("=" * 80)
    print("LOADING NEPALI V7 MODEL")
    print("=" * 80)

    model = NepaliGPT().to(DEVICE)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    state_dict = state_dict.copy()

    for key in list(state_dict.keys()):
        if "causal_mask" in key:
            del state_dict[key]

    model.load_state_dict(state_dict, strict=False)
    model.eval()
    print("[OK] Nepali V7 checkpoint loaded successfully.")

    test_data = []
    with open(TEST_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                test_data.append(json.loads(line.strip()))

    print(f"Test examples loaded: {len(test_data):,}")

    first_token_correct = 0
    exact_match_correct = 0
    total = 0

    task_correct = defaultdict(int)
    task_total = defaultdict(int)
    position_correct = defaultdict(int)
    position_total = defaultdict(int)
    failures = []

    for example in tqdm(test_data, desc="Evaluating"):
        prompt = example["prompt"]
        gold = example["target"].strip()
        task = example.get("task", "unknown")

        gold_ids = sp.encode_as_ids(gold)
        if len(gold_ids) == 0:
            continue

        pred_first_id = predict_first_token(model, prompt)
        first_correct = (pred_first_id == gold_ids[0])

        pred = generate_answer(model, prompt)
        exact_correct = (pred.strip() == gold)

        total += 1
        first_token_correct += int(first_correct)
        exact_match_correct += int(exact_correct)

        task_total[task] += 1
        task_correct[task] += int(exact_correct)

        position = example.get("answer_pos", example.get("answer_position"))
        if position is not None:
            key = f"pos{position}"
            position_total[key] += 1
            position_correct[key] += int(exact_correct)

        if not exact_correct and len(failures) < 20:
            failures.append({
                "task": task,
                "position": position,
                "prompt": prompt,
                "gold": gold,
                "pred": pred
            })

    first_token_accuracy = 100.0 * first_token_correct / total if total else 0.0
    exact_match_accuracy = 100.0 * exact_match_correct / total if total else 0.0

    print()
    print("=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    print(f"First-Token Accuracy : {first_token_correct}/{total} ({first_token_accuracy:.2f}%)")
    print(f"Exact Match Accuracy : {exact_match_correct}/{total} ({exact_match_accuracy:.2f}%)")

    print("\n--- Accuracy by Task ---")
    for task in sorted(task_total):
        correct = task_correct[task]
        count = task_total[task]
        acc = 100.0 * correct / count
        print(f"{task:<30}: {correct}/{count} ({acc:.2f}%)")

    if position_total:
        print("\n--- Accuracy by Answer Position ---")
        for key in sorted(position_total):
            correct = position_correct[key]
            count = position_total[key]
            acc = 100.0 * correct / count
            print(f"{key:<8}: {correct}/{count} ({acc:.2f}%)")

    if failures:
        print("\n" + "=" * 80)
        print("SAMPLE FAILURES (FIRST 5)")
        print("=" * 80)
        for i, failure in enumerate(failures[:5], 1):
            print(f"\n[FAIL {i}] Task: {failure['task']}")
            print("Prompt:")
            print(failure["prompt"])
            print(f"Gold: {failure['gold']}")
            print(f"Pred: {failure['pred']}")

    print("\n" + "=" * 80)
    print("NEPALI V7 EVALUATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    load_and_eval()
