import os
import sys
import math
import time
import torch

# Ensure package imports work when executed directly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from bonus.config import (
    SP_MODEL_PATH,
    TRAIN_TXT_PATH,
    VAL_TXT_PATH,
    TRAIN_BIN_PATH,
    VAL_BIN_PATH,
    CKPT_DIR,
    MODEL_CONFIG,
    TRAIN_CONFIG
)
from bonus.model import GPT
from bonus.train.dataset import tokenize_corpus_to_bin, compute_bytes_per_token, MemmapDataset

def make_scheduler(optimizer, warmup_steps, max_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

@torch.no_grad()
def evaluate_loss(model, val_ds, batch_size, device, amp_dtype, n_batches=20):
    model.eval()
    losses = []
    for _ in range(n_batches):
        x, y = val_ds.get_batch(batch_size, device)
        with torch.autocast(device_type=device, dtype=amp_dtype, enabled=(device != "cpu")):
            _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    mean_loss = sum(losses) / len(losses)
    return mean_loss, math.exp(mean_loss)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(CKPT_DIR, exist_ok=True)

    print(f"🚀 Initializing Ablated Model H (No Positional Embeddings) on {device}...")
    model = GPT(**MODEL_CONFIG).to(device)
    print(f"Total Parameters: {model.num_params():,} | Positional Embeddings: Disabled (None)")

    # Prepare datasets
    if not os.path.exists(TRAIN_BIN_PATH):
        print(f"Tokenizing {TRAIN_TXT_PATH} -> {TRAIN_BIN_PATH}...")
        tokenize_corpus_to_bin(TRAIN_TXT_PATH, SP_MODEL_PATH, TRAIN_BIN_PATH)
    else:
        print(f"Using pre-tokenized train dataset: {TRAIN_BIN_PATH}")

    if not os.path.exists(VAL_BIN_PATH):
        print(f"Tokenizing {VAL_TXT_PATH} -> {VAL_BIN_PATH}...")
        tokenize_corpus_to_bin(VAL_TXT_PATH, SP_MODEL_PATH, VAL_BIN_PATH)
    else:
        print(f"Using pre-tokenized val dataset: {VAL_BIN_PATH}")

    bytes_per_token = compute_bytes_per_token(VAL_TXT_PATH, SP_MODEL_PATH)
    print(f"Bytes / token (val): {bytes_per_token:.4f}")

    train_ds = MemmapDataset(TRAIN_BIN_PATH, MODEL_CONFIG["max_seq_len"])
    val_ds = MemmapDataset(VAL_BIN_PATH, MODEL_CONFIG["max_seq_len"])

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=TRAIN_CONFIG["lr"],
        betas=TRAIN_CONFIG["betas"],
        weight_decay=TRAIN_CONFIG["weight_decay"]
    )
    scheduler = make_scheduler(optimizer, TRAIN_CONFIG["warmup_steps"], TRAIN_CONFIG["max_steps"])
    amp_dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float16
    )

    best_val_loss = float("inf")
    t0 = time.time()
    model.train()

    print("\n🚀 Starting Pretraining Loop...")
    for step in range(TRAIN_CONFIG["max_steps"]):
        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0

        for _ in range(TRAIN_CONFIG["grad_accum_steps"]):
            x, y = train_ds.get_batch(TRAIN_CONFIG["batch_size"], device)
            with torch.autocast(device_type=device, dtype=amp_dtype, enabled=(device != "cpu")):
                _, loss = model(x, y)
                loss = loss / TRAIN_CONFIG["grad_accum_steps"]
            loss.backward()
            accum_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), TRAIN_CONFIG["grad_clip_norm"])
        optimizer.step()
        scheduler.step()

        if step % 50 == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - t0
            print(f"step {step:7d} | loss {accum_loss:.4f} | lr {current_lr:.2e} | {elapsed:.0f}s")

        if step % TRAIN_CONFIG["eval_interval"] == 0:
            val_loss, ppl = evaluate_loss(model, val_ds, TRAIN_CONFIG["batch_size"], device, amp_dtype)
            bpb = val_loss / math.log(2) / bytes_per_token
            print(f"  [eval] step {step} | val_loss {val_loss:.4f} | ppl {ppl:.2f} | bpb {bpb:.3f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_path = os.path.join(CKPT_DIR, "best.pt")
                torch.save(model.state_dict(), best_path)
                print(f"  [ckpt] new best (val_loss {val_loss:.4f}) saved to best.pt")

            latest_path = os.path.join(CKPT_DIR, "latest.pt")
            torch.save({
                "step": step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_loss": best_val_loss
            }, latest_path)

    print(f"\n✅ Training Complete! Best Validation Loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    main()
