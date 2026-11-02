import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import sentencepiece as spm
from tqdm import tqdm
import json
import os
import sys
import math
import time
from collections import defaultdict

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# =============================================================================
# HARD TRAINING — HARD DATA FINE-TUNING FROM SOFT (NEPALI)
# =============================================================================

print("=" * 80)
print("HARD TRAINING — HARD DATA FROM SOFT (NEPALI)")
print("=" * 80)

# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def resolve_path(env_key, default_name):
    env_val = os.getenv(env_key)
    local_p = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data", default_name))
    if os.name == "nt":
        if env_val and os.path.exists(env_val):
            return env_val
        return local_p
    return env_val or local_p

TOKENIZER_PATH = resolve_path("TOKENIZER_PATH", "nepali_bpe_32000.model")
SOFT_CHECKPOINT = resolve_path("SOFT_CHECKPOINT", "soft_best.pt")
TRAIN_PATH = resolve_path("TRAIN_PATH", "hard_train.jsonl")
VAL_PATH = resolve_path("VAL_PATH", "hard_val.jsonl")
SAVE_PATH = resolve_path("SAVE_PATH", "hard_best.pt")

BATCH_SIZE = 32
LR = 1e-5
EPOCHS = 3

FIRST_TOKEN_WEIGHT = 3.0
WARMUP_RATIO = 0.06
WEIGHT_DECAY = 0.01

MAX_SEQ_LEN = 512

EVAL_INTERVAL = 300
ACC_EVAL_SAMPLES = 600
PATIENCE = 5

print(f"Learning rate       : {LR}")
print(f"Epochs              : {EPOCHS}")
print(f"First-token weight  : {FIRST_TOKEN_WEIGHT}")
print(f"Warmup ratio        : {WARMUP_RATIO}")
print(f"Weight decay        : {WEIGHT_DECAY}")
print(f"Batch size          : {BATCH_SIZE}")
print(f"Device              : {DEVICE}")
print(f"Tokenizer path      : {TOKENIZER_PATH}")
print(f"Soft Checkpoint     : {SOFT_CHECKPOINT}")
print()

# =============================================================================
# TOKENIZER
# =============================================================================

if not os.path.exists(TOKENIZER_PATH):
    raise FileNotFoundError(f"Tokenizer not found: {TOKENIZER_PATH}")

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
# MODEL
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

        q = q.view(
            B, T, self.nhead, self.head_dim
        ).transpose(1, 2)

        k = k.view(
            B, T, self.nhead, self.head_dim
        ).transpose(1, 2)

        v = v.view(
            B, T, self.nhead, self.head_dim
        ).transpose(1, 2)

        attn_mask = self.causal_mask[:, :, :T, :T]

        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=~attn_mask
        )

        y = y.transpose(1, 2).contiguous()
        y = y.view(B, T, C)

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

        self.attn = CausalSelfAttention(
            d_model,
            nhead,
            max_seq_len
        )

        self.ln2 = nn.LayerNorm(d_model)

        self.ffn = nn.ModuleDict({
            "fc1": nn.Linear(d_model, dim_feedforward),
            "fc2": nn.Linear(dim_feedforward, d_model)
        })

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        x = x + self.dropout(
            self.attn(self.ln1(x))
        )

        x = x + self.dropout(
            self.ffn["fc2"](
                F.gelu(
                    self.ffn["fc1"](
                        self.ln2(x)
                    )
                )
            )
        )

        return x


class NepaliGPT(nn.Module):

    def __init__(self):

        super().__init__()

        self.vocab_size = VOCAB_SIZE
        self.d_model = 448
        self.max_seq_len = MAX_SEQ_LEN

        self.tok_emb = nn.Embedding(
            self.vocab_size,
            self.d_model
        )

        self.pos_emb = nn.Embedding(
            self.max_seq_len,
            self.d_model
        )

        self.drop = nn.Dropout(0.1)

        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=448,
                nhead=8,
                dim_feedforward=1024,
                max_seq_len=MAX_SEQ_LEN,
                dropout=0.1
            )
            for _ in range(6)
        ])

        self.ln_f = nn.LayerNorm(self.d_model)

        self.lm_head = nn.Linear(
            self.d_model,
            self.vocab_size,
            bias=False
        )

        self.lm_head.weight = self.tok_emb.weight

    def forward(self, idx):

        B, T = idx.shape

        if T > self.max_seq_len:
            raise ValueError(
                f"Sequence length {T} exceeds "
                f"max_seq_len {self.max_seq_len}"
            )

        pos = torch.arange(
            T,
            device=idx.device
        )

        x = (
            self.tok_emb(idx)
            + self.pos_emb(pos)
        )

        x = self.drop(x)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)

        return self.lm_head(x)


HindiGPT = NepaliGPT

# =============================================================================
# DATASET
# =============================================================================

class HardDataset(Dataset):

    def __init__(self, filename):

        self.data = []

        if not os.path.exists(filename):
            print(f"Warning: Data file not found: {filename}")
            return

        with open(
            filename,
            "r",
            encoding="utf-8"
        ) as f:

            for line in f:

                d = json.loads(line)

                prompt = d["prompt"]
                target = d["target"]

                prompt_ids = sp.encode_as_ids(prompt)

                target_ids = (
                    sp.encode_as_ids(target)
                    + [EOS_ID]
                )

                input_ids = prompt_ids + target_ids

                labels = (
                    [-100] * len(prompt_ids)
                    + target_ids
                )

                weights = (
                    [0.0] * len(prompt_ids)
                    + [FIRST_TOKEN_WEIGHT]
                    + [1.0] * (len(target_ids) - 1)
                )

                if len(input_ids) > MAX_SEQ_LEN:
                    continue

                self.data.append({
                    "input_ids": torch.tensor(
                        input_ids,
                        dtype=torch.long
                    ),
                    "labels": torch.tensor(
                        labels,
                        dtype=torch.long
                    ),
                    "weights": torch.tensor(
                        weights,
                        dtype=torch.float
                    ),
                    "task": d.get("task", "unknown")
                })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn(batch):

    max_len = max(
        len(x["input_ids"])
        for x in batch
    )

    B = len(batch)

    inputs = torch.full(
        (B, max_len),
        PAD_ID,
        dtype=torch.long
    )

    labels = torch.full(
        (B, max_len),
        -100,
        dtype=torch.long
    )

    weights = torch.zeros(
        (B, max_len),
        dtype=torch.float
    )

    tasks = []

    for i, item in enumerate(batch):

        n = len(item["input_ids"])

        inputs[i, :n] = item["input_ids"]
        labels[i, :n] = item["labels"]
        weights[i, :n] = item["weights"]

        tasks.append(item["task"])

    return inputs, labels, weights, tasks


# =============================================================================
# ACCURACY EVALUATION
# =============================================================================

@torch.no_grad()
def evaluate_accuracy(
    dataset,
    model,
    device,
    num_samples=600
):

    model.eval()

    correct = 0
    total = 0

    task_correct = defaultdict(int)
    task_total = defaultdict(int)

    limit = min(
        num_samples,
        len(dataset)
    )

    indices = list(range(limit))

    for idx in indices:

        item = dataset[idx]

        prompt_ids = item["input_ids"][
            item["labels"] == -100
        ].tolist()

        target_ids = item["labels"][
            item["labels"] != -100
        ].tolist()

        if len(prompt_ids) == 0:
            continue

        prompt = torch.tensor(
            [prompt_ids],
            dtype=torch.long,
            device=device
        )

        logits = model(prompt)

        pred = logits[
            0,
            -1
        ].argmax().item()

        gold = target_ids[0]

        is_correct = pred == gold

        total += 1
        correct += int(is_correct)

        task = item["task"]

        task_total[task] += 1
        task_correct[task] += int(is_correct)

    model.train()

    overall = (
        100.0 * correct / total
        if total
        else 0.0
    )

    task_acc = {}

    for task in sorted(task_total):

        task_acc[task] = (
            100.0
            * task_correct[task]
            / task_total[task]
        )

    return overall, task_acc


def run_training():
    if not os.path.exists(TRAIN_PATH) or not os.path.exists(VAL_PATH):
        print("Training or validation dataset not found. Exiting.")
        return

    train_dataset = HardDataset(TRAIN_PATH)
    val_dataset = HardDataset(VAL_PATH)

    print("Dataset:")
    print(f"Train samples: {len(train_dataset):,}")
    print(f"Val samples  : {len(val_dataset):,}")
    print()

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=True
    )

    model = NepaliGPT().to(DEVICE)

    if os.path.exists(SOFT_CHECKPOINT):
        print("=" * 80)
        print("LOADING SOFT CHECKPOINT")
        print("=" * 80)

        checkpoint = torch.load(SOFT_CHECKPOINT, map_location=DEVICE)
        state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
        state_dict = state_dict.copy()

        for key in list(state_dict.keys()):
            if "causal_mask" in key:
                del state_dict[key]

        model.load_state_dict(state_dict, strict=False)
        print("[OK] SOFT checkpoint loaded successfully.")
    else:
        print(f"Warning: SOFT checkpoint missing at {SOFT_CHECKPOINT}. Initializing randomly.")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    total_steps = len(train_loader) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    loss_fct = nn.CrossEntropyLoss(reduction="none")

    best_acc = -1.0
    best_step = 0
    patience_count = 0
    global_step = 0
    start_time = time.time()

    print("=" * 80)
    print("STARTING HARD TRAINING (NEPALI)")
    print("=" * 80)

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}")
        for inputs, labels, weights, tasks in pbar:
            inputs = inputs.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            weights = weights.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_weights = weights[..., 1:].contiguous()

            loss = loss_fct(
                shift_logits.view(-1, model.vocab_size),
                shift_labels.view(-1)
            )

            active = shift_labels.view(-1) != -100
            weighted = loss * shift_weights.view(-1)
            active_tokens = active.sum()

            final_loss = weighted.sum() / active_tokens.clamp(min=1)
            final_loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            global_step += 1
            epoch_loss += final_loss.item()
            current_lr = optimizer.param_groups[0]["lr"]

            pbar.set_postfix(loss=f"{final_loss.item():.4f}", lr=f"{current_lr:.2e}")

            if global_step % EVAL_INTERVAL == 0:
                acc, task_acc = evaluate_accuracy(val_dataset, model, DEVICE, ACC_EVAL_SAMPLES)
                elapsed = time.time() - start_time
                print()
                print(f"[EVAL] Step {global_step} | Val First-Token Acc: {acc:.2f}% | Elapsed: {elapsed:.0f}s")
                for t, v in task_acc.items():
                    print(f"  {t:<28}: {v:.2f}%")

                if acc > best_acc:
                    best_acc = acc
                    best_step = global_step
                    patience_count = 0
                    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
                    torch.save({
                        "model": model.state_dict(),
                        "best_accuracy": best_acc,
                        "best_step": best_step
                    }, SAVE_PATH)
                    print(f">>> NEW BEST HARD CHECKPOINT SAVED ({best_acc:.2f}%)")
                else:
                    patience_count += 1
                    print(f"[Patience] {patience_count}/{PATIENCE}")
                    if patience_count >= PATIENCE:
                        print("\nEarly stopping.")
                        break

        if patience_count >= PATIENCE:
            break

    print("=" * 80)
    print("HARD TRAINING COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    run_training()
