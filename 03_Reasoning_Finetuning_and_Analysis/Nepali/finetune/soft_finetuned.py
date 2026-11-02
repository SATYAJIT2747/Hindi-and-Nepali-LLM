# ==============================================================================
# SOFT TRAINING — NEPALI FULL TRAINING RECIPE
# ==============================================================================

import os
import sys
import math
import json
import time
import random
from collections import defaultdict

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import torch
import torch.nn as nn
import torch.nn.functional as F
import sentencepiece as spm
from torch.utils.data import Dataset, DataLoader


# ==============================================================================
# 1. REPRODUCIBILITY
# ==============================================================================

random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


# ==============================================================================
# 2. CONFIG
# ==============================================================================

CONFIG = {
    "vocab_size": 32000,
    "d_model": 448,
    "n_layers": 6,
    "n_heads": 8,
    "d_ff": 1024,
    "max_seq_len": 512,
    "dropout": 0.1,

    "lr": 2e-5,
    "weight_decay": 0.01,
    "batch_size": 32,
    "epochs": 3,

    "first_token_weight": 3.0,

    "eval_interval": 300,
    "patience": 5,
    "acc_eval_subset": 600,

    "device": "cuda" if torch.cuda.is_available() else "cpu",

    "amp_dtype":
        torch.bfloat16
        if (torch.cuda.is_available() and torch.cuda.is_bf16_supported())
        else torch.float16,
}


# ==============================================================================
# 3. PATHS
# ==============================================================================

def resolve_path(env_key, default_name):
    env_val = os.getenv(env_key)
    local_p = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data", default_name))
    if os.name == "nt":
        if env_val and os.path.exists(env_val):
            return env_val
        return local_p
    return env_val or local_p

TOKENIZER_PATH = resolve_path("TOKENIZER_PATH", "nepali_bpe_32000.model")
BASE_CHECKPOINT = resolve_path("BASE_CHECKPOINT", "nepali_gpt.pt")
TRAIN_PATH = resolve_path("TRAIN_PATH", "soft_train.jsonl")
VAL_PATH = resolve_path("VAL_PATH", "soft_val.jsonl")
SAVE_PATH = resolve_path("SAVE_PATH", "soft_best.pt")


print("=" * 80)
print("SOFT TRAINING (NEPALI)")
print("=" * 80)
print(f"Learning rate       : {CONFIG['lr']}")
print(f"Epochs              : {CONFIG['epochs']}")
print(f"First-token weight  : {CONFIG['first_token_weight']}")
print(f"Weight decay        : {CONFIG['weight_decay']}")
print(f"Batch size          : {CONFIG['batch_size']}")
print(f"Device              : {CONFIG['device']}")
print(f"Tokenizer           : {TOKENIZER_PATH}")
print(f"Base Checkpoint     : {BASE_CHECKPOINT}")
print()


# ==============================================================================
# 4. TOKENIZER
# ==============================================================================

if not os.path.exists(TOKENIZER_PATH):
    raise FileNotFoundError(f"Tokenizer not found: {TOKENIZER_PATH}")

sp = spm.SentencePieceProcessor(model_file=TOKENIZER_PATH)
EOS_ID = sp.eos_id() if sp.eos_id() >= 0 else 2
PAD_ID = sp.pad_id() if sp.pad_id() >= 0 else 3

print("Tokenizer loaded.")
print(f"EOS ID: {EOS_ID}")
print(f"PAD ID: {PAD_ID}")
print()


# ==============================================================================
# 5. MODEL ARCHITECTURE
# ==============================================================================

class CausalSelfAttention(nn.Module):

    def __init__(self, d_model, n_heads, max_seq_len, dropout):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

        mask = torch.tril(
            torch.ones(max_seq_len, max_seq_len)
        ).view(1, 1, max_seq_len, max_seq_len)

        self.register_buffer("causal_mask", mask)

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(x).split(D, dim=2)

        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.d_head))
        att = att.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, D)
        return self.resid_drop(self.out_proj(y))


class FeedForward(nn.Module):

    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.fc2(self.act(self.fc1(x))))


class Block(nn.Module):

    def __init__(self, d_model, n_heads, d_ff, max_seq_len, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, max_seq_len, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class NepaliGPT(nn.Module):

    def __init__(
        self,
        vocab_size=32000,
        d_model=448,
        n_layers=6,
        n_heads=8,
        d_ff=1024,
        max_seq_len=512,
        dropout=0.1
    ):
        super().__init__()
        self.max_seq_len = max_seq_len

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.emb_drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            Block(d_model, n_heads, d_ff, max_seq_len, dropout)
            for _ in range(n_layers)
        ])

        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx):
        B, T = idx.shape
        assert T <= self.max_seq_len

        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        x = self.emb_drop(self.tok_emb(idx) + self.pos_emb(pos))

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        return self.lm_head(x)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())


GPT = NepaliGPT
HindiGPT = NepaliGPT


# ==============================================================================
# 6. DATASET & LOSS
# ==============================================================================

class NepaliReasoningDataset(Dataset):

    def __init__(self, jsonl_path, tokenizer, max_len=512):
        self.samples = []
        if os.path.exists(jsonl_path):
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        self.samples.append(json.loads(line.strip()))
        self.sp = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        prompt_ids = self.sp.encode(item["prompt"], out_type=int)
        target_ids = self.sp.encode(item["target"], out_type=int) + [EOS_ID]

        input_ids = prompt_ids + target_ids
        labels = [-100] * len(prompt_ids) + target_ids

        if len(input_ids) > self.max_len:
            input_ids = input_ids[:self.max_len]
            labels = labels[:self.max_len]

        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long)
        )


def collate_fn(batch):
    input_list, label_list = zip(*batch)
    max_len = max(len(x) for x in input_list)

    padded_inputs, padded_labels = [], []
    for x, y in zip(input_list, label_list):
        pad_len = max_len - len(x)
        padded_inputs.append(torch.cat([x, torch.full((pad_len,), PAD_ID, dtype=torch.long)]))
        padded_labels.append(torch.cat([y, torch.full((pad_len,), -100, dtype=torch.long)]))

    return torch.stack(padded_inputs), torch.stack(padded_labels)


def weighted_loss(logits, labels, first_token_weight=1.0):
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    B, T, V = shift_logits.shape

    loss_flat = F.cross_entropy(
        shift_logits.view(-1, V),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="none"
    ).view(B, T)

    valid_mask = (shift_labels != -100)
    weights = valid_mask.float()

    if first_token_weight != 1.0:
        for b in range(B):
            valid_idx = valid_mask[b].nonzero(as_tuple=True)[0]
            if len(valid_idx) > 0:
                weights[b, valid_idx[0]] = first_token_weight

    total_w = weights.sum().clamp(min=1.0)
    return (loss_flat * weights).sum() / total_w


@torch.no_grad()
def greedy_generate(model, prompt_ids, device, max_new_tokens=8):
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    generated = []
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.max_seq_len:]
        logits = model(idx_cond)
        next_id = torch.argmax(logits[:, -1, :], dim=-1).item()
        if next_id == EOS_ID:
            break
        generated.append(next_id)
        next_tensor = torch.tensor([[next_id]], dtype=torch.long, device=device)
        idx = torch.cat([idx, next_tensor], dim=1)
    return sp.decode(generated).strip()


@torch.no_grad()
def evaluate_loss(loader, model, device, amp_dtype):
    model.eval()
    total = 0.0
    n = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        with torch.autocast(device_type="cuda" if "cuda" in device else "cpu", dtype=amp_dtype):
            logits = model(x)
            loss = weighted_loss(logits, y, first_token_weight=1.0)
        total += loss.item()
        n += 1
    model.train()
    mean_loss = total / max(1, n)
    ppl = math.exp(min(mean_loss, 20))
    return mean_loss, ppl


@torch.no_grad()
def evaluate_accuracy(subset, model, device):
    model.eval()
    correct = 0
    first_tok_correct = 0
    task_correct = defaultdict(int)
    task_total = defaultdict(int)

    for item in subset:
        prompt_ids = sp.encode(item["prompt"], out_type=int)
        gold = item["target"].strip()
        gold_ids = sp.encode(gold, out_type=int)
        task = item.get("task_type", item.get("task", "unknown"))

        idx_t = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        logits = model(idx_t)
        top_id = torch.argmax(logits[0, -1, :], dim=-1).item()

        if gold_ids and top_id == gold_ids[0]:
            first_tok_correct += 1

        pred = greedy_generate(model, prompt_ids, device)
        is_correct = (pred == gold)
        correct += int(is_correct)

        task_total[task] += 1
        task_correct[task] += int(is_correct)

    model.train()
    n = len(subset)
    if n == 0:
        return 0.0, 0.0, {}

    task_acc = {task: (task_correct[task] / task_total[task]) * 100 for task in task_total}
    return (correct / n * 100, first_tok_correct / n * 100, task_acc)


def run_training():
    if not os.path.exists(TRAIN_PATH) or not os.path.exists(VAL_PATH):
        print("Training or validation soft dataset missing. Exiting.")
        return

    train_dataset = NepaliReasoningDataset(TRAIN_PATH, sp, CONFIG["max_seq_len"])
    val_dataset = NepaliReasoningDataset(VAL_PATH, sp, CONFIG["max_seq_len"])

    print("Dataset:")
    print(f"Train samples: {len(train_dataset):,}")
    print(f"Val samples  : {len(val_dataset):,}")
    print()

    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=True
    )

    acc_subset = random.sample(
        val_dataset.samples,
        min(CONFIG["acc_eval_subset"], len(val_dataset.samples))
    )

    model = NepaliGPT(
        vocab_size=CONFIG["vocab_size"],
        d_model=CONFIG["d_model"],
        n_layers=CONFIG["n_layers"],
        n_heads=CONFIG["n_heads"],
        d_ff=CONFIG["d_ff"],
        max_seq_len=CONFIG["max_seq_len"],
        dropout=CONFIG["dropout"]
    ).to(CONFIG["device"])

    if os.path.exists(BASE_CHECKPOINT):
        print(f"Loading base checkpoint from {BASE_CHECKPOINT}...")
        checkpoint = torch.load(BASE_CHECKPOINT, map_location=CONFIG["device"])
        state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
        model.load_state_dict(state_dict, strict=False)
        print(f"[OK] Loaded base checkpoint with {model.num_params():,} parameters.")
    else:
        print(f"Warning: Base checkpoint missing at {BASE_CHECKPOINT}. Initializing randomly.")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG["lr"],
        betas=(0.9, 0.98),
        weight_decay=CONFIG["weight_decay"]
    )

    total_steps = len(train_loader) * CONFIG["epochs"]
    warmup_steps = int(0.06 * total_steps)

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_acc = -1.0
    global_step = 0
    patience_counter = 0
    early_stop = False
    t0 = time.time()
    running_loss = 0.0

    print("=" * 80)
    print("STARTING NEPALI SOFT TRAINING")
    print("=" * 80)

    for epoch in range(1, CONFIG["epochs"] + 1):
        if early_stop:
            break
        model.train()

        for x, y in train_loader:
            x = x.to(CONFIG["device"], non_blocking=True)
            y = y.to(CONFIG["device"], non_blocking=True)

            with torch.autocast(
                device_type="cuda" if "cuda" in CONFIG["device"] else "cpu",
                dtype=CONFIG["amp_dtype"]
            ):
                logits = model(x)
                loss = weighted_loss(logits, y, first_token_weight=CONFIG["first_token_weight"])

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            running_loss += loss.item()
            global_step += 1

            if global_step % 50 == 0:
                avg_loss = running_loss / 50
                current_lr = scheduler.get_last_lr()[0]
                elapsed = time.time() - t0
                print(f"Epoch {epoch} | Step {global_step}/{total_steps} | Train Loss: {avg_loss:.4f} | LR: {current_lr:.2e} | Elapsed: {elapsed:.0f}s")
                running_loss = 0.0

            if global_step % CONFIG["eval_interval"] == 0:
                val_loss, val_ppl = evaluate_loss(val_loader, model, CONFIG["device"], CONFIG["amp_dtype"])
                acc, first_tok_acc, task_acc = evaluate_accuracy(acc_subset, model, CONFIG["device"])
                print(f"\n[EVAL] Step {global_step} | Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:.2f} | Reasoning Acc: {acc:.2f}% | First-Token Acc: {first_tok_acc:.2f}%")
                if task_acc:
                    print("Task-wise: " + ", ".join(f"{t}={a:.1f}%" for t, a in task_acc.items()))

                if acc > best_acc:
                    best_acc = acc
                    patience_counter = 0
                    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
                    torch.save({"model": model.state_dict(), "config": CONFIG, "best_accuracy": best_acc}, SAVE_PATH)
                    print(f">>> New best checkpoint saved (Reasoning Acc: {best_acc:.2f}%)")
                else:
                    patience_counter += 1
                    print(f"[Patience] {patience_counter}/{CONFIG['patience']} evals without accuracy improvement.")
                    if patience_counter >= CONFIG["patience"]:
                        print("Early stopping triggered.")
                        early_stop = True
                        break

    print("=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    run_training()
