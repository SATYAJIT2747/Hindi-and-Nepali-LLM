"""Tiny YAML config loader shared across train/ and eval/ scripts."""
import yaml
import torch

_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # normalize a couple of fields that need to become real torch objects
    if "amp_dtype" in cfg and isinstance(cfg["amp_dtype"], str):
        cfg["amp_dtype"] = _DTYPES[cfg["amp_dtype"]]
    if "betas" in cfg:
        cfg["betas"] = tuple(cfg["betas"])
    return cfg
