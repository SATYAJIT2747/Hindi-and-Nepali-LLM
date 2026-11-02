"""
Full generation-quality evaluation for a trained model: greedy, temperature
sampling (0.5/1.0/1.5), top-k (20/50), nucleus/top-p (0.8/0.9/0.95), and beam
search (width 4) - each scored with BLEU-4 / chrF / chrF++ / ROUGE-L plus
repetition-rate / Distinct-1 / Distinct-2 diagnostics.

Usage:
    python -m eval.evaluate --config configs/hindi.yaml --checkpoint checkpoints/hindi/best.pt
"""
import argparse
import json
import os
import sys

import sentencepiece as spm
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs import load_config
from model import build_model
from train.checkpoint import load_model_only
from eval.generate import generate, generate_topk_topp, generate_beam_search, build_holdout_examples
from eval.metrics import compute_metrics, print_metrics


def run_setting(model, holdout, decode_fn, sp, device):
    hyps, refs = [], []
    for ex in holdout:
        gen_ids = decode_fn(model, ex["prefix_ids"], max_new_tokens=len(ex["ref_ids"]), device=device)
        gen_ids = gen_ids[len(ex["prefix_ids"]):]
        hyps.append(sp.decode(gen_ids))
        refs.append(ex["ref_text"])
    return hyps, refs


def main(config_path, checkpoint_path, n_examples, out_json):
    cfg = load_config(config_path)
    model = build_model(cfg).to(cfg["device"])
    load_model_only(checkpoint_path, model, cfg["device"])
    model.eval()
    print(f"Model ready: {model.num_params():,} params (checkpoint: {checkpoint_path})")

    sp = spm.SentencePieceProcessor(model_file=cfg["sp_model"])
    device = cfg["device"]

    holdout = build_holdout_examples(cfg["val_txt"], sp, n_examples=n_examples)

    settings = {
        "greedy":     lambda m, p, max_new_tokens, device: generate(m, p, max_new_tokens, greedy=True, device=device),
        "temp_0.5":   lambda m, p, max_new_tokens, device: generate(m, p, max_new_tokens, temperature=0.5, device=device),
        "temp_1.0":   lambda m, p, max_new_tokens, device: generate(m, p, max_new_tokens, temperature=1.0, device=device),
        "temp_1.5":   lambda m, p, max_new_tokens, device: generate(m, p, max_new_tokens, temperature=1.5, device=device),
        "top_k_20":   lambda m, p, max_new_tokens, device: generate_topk_topp(m, p, max_new_tokens, top_k=20, device=device),
        "top_k_50":   lambda m, p, max_new_tokens, device: generate_topk_topp(m, p, max_new_tokens, top_k=50, device=device),
        "top_p_0.8":  lambda m, p, max_new_tokens, device: generate_topk_topp(m, p, max_new_tokens, top_p=0.8, device=device),
        "top_p_0.9":  lambda m, p, max_new_tokens, device: generate_topk_topp(m, p, max_new_tokens, top_p=0.9, device=device),
        "top_p_0.95": lambda m, p, max_new_tokens, device: generate_topk_topp(m, p, max_new_tokens, top_p=0.95, device=device),
        "beam_4":     lambda m, p, max_new_tokens, device: generate_beam_search(m, p, max_new_tokens, num_beams=4, device=device),
    }

    all_metrics = {}
    all_samples = {}
    for name, decode_fn in settings.items():
        hyps, refs = run_setting(model, holdout, decode_fn, sp, device)
        metrics = compute_metrics(hyps, refs)
        print_metrics(name, metrics)
        all_metrics[name] = metrics
        all_samples[name] = [
            {"prefix": ex["prefix_text"], "ref": ex["ref_text"], "gen": h}
            for ex, h in zip(holdout[:3], hyps[:3])
        ]

    if out_json:
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump({"metrics": all_metrics, "samples": all_samples}, f, ensure_ascii=False, indent=2)
        print(f"\nSaved full results -> {out_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--n-examples", type=int, default=50)
    parser.add_argument("--out-json", default=None)
    args = parser.parse_args()
    main(args.config, args.checkpoint, args.n_examples, args.out_json)
