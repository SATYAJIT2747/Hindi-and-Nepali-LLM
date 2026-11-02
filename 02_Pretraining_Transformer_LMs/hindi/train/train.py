"""
Pretrain a decoder-only GPT model on a single language's monolingual corpus.

Usage:
    python -m train.train --config configs/hindi.yaml
    python -m train.train --config configs/nepali.yaml
"""
import argparse
import json
import math
import os
import time
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs import load_config
from model import build_model
from train.data import MemmapDataset, ensure_tokenized
from train.checkpoint import save_checkpoint, load_checkpoint, find_resumable_checkpoint


def make_scheduler(optimizer, cfg):
    def lr_lambda(step):
        if step < cfg["warmup_steps"]:
            return step / max(1, cfg["warmup_steps"])
        progress = (step - cfg["warmup_steps"]) / max(1, cfg["max_steps"] - cfg["warmup_steps"])
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


@torch.no_grad()
def evaluate(model, val_ds, cfg, n_batches=20):
    model.eval()
    losses = []
    for _ in range(n_batches):
        x, y = val_ds.get_batch(cfg["batch_size"], cfg["device"])
        with torch.autocast(device_type=cfg["device"], dtype=cfg["amp_dtype"], enabled=(cfg["device"] != "cpu")):
            _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    mean_loss = sum(losses) / len(losses)
    return mean_loss, math.exp(mean_loss)


def main(config_path, resume_search_root=None):
    cfg = load_config(config_path)
    bytes_per_token = ensure_tokenized(cfg)
    print(f"[{cfg['language']}] bytes/token (val): {bytes_per_token:.4f}")

    model = build_model(cfg).to(cfg["device"])
    print(f"[{cfg['language']}] params: {model.num_params():,} "
          f"(non-embedding: {model.num_params(non_embedding=True):,})")

    # sanity check required by the assignment - verify the causal mask blocks the future
    dummy = torch.randint(0, cfg["vocab_size"], (2, 16), device=cfg["device"])
    model.causality_check(dummy)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["lr"], betas=tuple(cfg["betas"]), weight_decay=cfg["weight_decay"]
    )
    scheduler = make_scheduler(optimizer, cfg)

    train_ds = MemmapDataset(cfg["train_bin"], cfg["max_seq_len"])
    val_ds = MemmapDataset(cfg["val_bin"], cfg["max_seq_len"])

    start_step, best_val_loss = 0, float("inf")
    resumable = find_resumable_checkpoint(cfg["ckpt_dir"], "latest.pt", search_root=resume_search_root)
    if resumable:
        start_step, best_val_loss = load_checkpoint(resumable, model, optimizer, scheduler, cfg["device"])
        print(f"Resumed from step {start_step} (best val_loss so far: {best_val_loss:.4f})")
    else:
        print("No checkpoint found - starting fresh")

    model.train()
    t0 = time.time()
    for step in range(start_step, cfg["max_steps"]):
        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0
        for _ in range(cfg["grad_accum_steps"]):
            x, y = train_ds.get_batch(cfg["batch_size"], cfg["device"])
            with torch.autocast(device_type=cfg["device"], dtype=cfg["amp_dtype"], enabled=(cfg["device"] != "cpu")):
                _, loss = model(x, y)
                loss = loss / cfg["grad_accum_steps"]
            loss.backward()
            accum_loss += loss.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip_norm"])
        optimizer.step()
        scheduler.step()

        if step % 50 == 0:
            print(f"step {step:6d} | loss {accum_loss:.4f} | lr {scheduler.get_last_lr()[0]:.2e} "
                  f"| {time.time() - t0:.0f}s")

        if step > 0 and step % cfg["eval_interval"] == 0:
            val_loss, ppl = evaluate(model, val_ds, cfg)
            bpb = val_loss / math.log(2) / bytes_per_token
            print(f"  [eval] step {step} | val_loss {val_loss:.4f} | ppl {ppl:.2f} | bpb {bpb:.3f}")

            save_checkpoint(cfg["latest_ckpt"], model, optimizer, scheduler, step, cfg,
                             extra={"best_val_loss": min(best_val_loss, val_loss)})
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(cfg["best_ckpt"], model, optimizer, scheduler, step, cfg,
                                 extra={"best_val_loss": best_val_loss})
                print(f"  [ckpt] new best (val_loss {best_val_loss:.4f}) saved to {cfg['best_ckpt']}")
            print(f"  [ckpt] latest saved at step {step}")

    save_checkpoint(cfg["latest_ckpt"], model, optimizer, scheduler, cfg["max_steps"], cfg,
                     extra={"best_val_loss": best_val_loss})
    final_val_loss, final_ppl = evaluate(model, val_ds, cfg, n_batches=50)
    final_bpb = final_val_loss / math.log(2) / bytes_per_token
    print("TRAINING DONE")
    print(f"final val_loss={final_val_loss:.4f} ppl={final_ppl:.2f} bpb={final_bpb:.3f} "
          f"params={model.num_params():,}")

    # Guard against max_steps being smaller than eval_interval (or eval_interval never being
    # hit) - in that edge case best_ckpt would otherwise never get written at all.
    if best_val_loss == float("inf") or not os.path.exists(cfg["best_ckpt"]):
        best_val_loss = final_val_loss
        save_checkpoint(cfg["best_ckpt"], model, optimizer, scheduler, cfg["max_steps"], cfg,
                         extra={"best_val_loss": best_val_loss})
        print(f"[ckpt] no mid-training eval improved best_ckpt - saved final model as best "
              f"(val_loss {best_val_loss:.4f}) -> {cfg['best_ckpt']}")
    else:
        print(f"best val_loss during training: {best_val_loss:.4f} -> saved at {cfg['best_ckpt']}")

    # pathway comparison record (matches the original notebook's pathway_result.json),
    # used to compare architecture pathways (e.g. shallow/wide vs deep/narrow) across runs
    pathway_name = cfg.get("pathway_name")
    if pathway_name:
        result_path = os.path.join(cfg["ckpt_dir"], "pathway_result.json")
        serializable_cfg = {k: v for k, v in cfg.items() if k not in ("device", "amp_dtype")}
        with open(result_path, "w") as f:
            json.dump({
                "pathway": pathway_name,
                "config": serializable_cfg,
                "final_val_loss": final_val_loss,
                "final_ppl": final_ppl,
                "final_bpb": final_bpb,
                "best_val_loss": best_val_loss,
                "params": model.num_params(),
            }, f, indent=2)
        print(f"Saved pathway comparison record -> {result_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to a configs/*.yaml file")
    parser.add_argument("--resume-search-root", default=None,
                         help="Optional extra directory tree to search for a resumable checkpoint")
    args = parser.parse_args()
    main(args.config, args.resume_search_root)
