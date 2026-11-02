"""Checkpoint save/load/resume. Every checkpoint stores model weights, optimizer
state, scheduler state, the current step, and the run's config dict."""
import os
import glob
import shutil
import torch


def save_checkpoint(path, model, optimizer, scheduler, step, config, extra=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "step": step,
        "config": config,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_checkpoint(path, model, optimizer, scheduler, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    return ckpt["step"], ckpt.get("best_val_loss", float("inf"))


def find_resumable_checkpoint(ckpt_dir, filename="latest.pt", search_root=None):
    """Check ckpt_dir first, then optionally search a wider root (e.g. a mounted
    'previous session output' directory) in case ckpt_dir itself was wiped."""
    local_path = os.path.join(ckpt_dir, filename)
    if os.path.exists(local_path):
        return local_path
    if search_root:
        candidates = glob.glob(f"{search_root}/**/{filename}", recursive=True)
        if candidates:
            os.makedirs(ckpt_dir, exist_ok=True)
            shutil.copy(candidates[0], local_path)
            print(f"Recovered checkpoint from a previous session output: {candidates[0]}")
            return local_path
    return None


def load_model_only(path, model, device):
    """Load just the model weights from a checkpoint (used at eval time)."""
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    return ckpt
