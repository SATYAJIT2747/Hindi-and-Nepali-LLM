"""
Attention analysis: capture per-head attention probabilities from a trained
model, plot query-vs-key heatmaps (layer 0 + last layer, first 4 heads), and
compute per-head entropy and mean attention distance.

Usage:
    python -m eval.attention_heatmap --config configs/hindi.yaml \
        --checkpoint checkpoints/hindi/best.pt --out hindi_attention_heatmap.png
"""
import argparse
import math
import os
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import sentencepiece as spm

# Try to find a font with Devanagari (or other non-Latin script) glyph coverage so
# token-label axes don't render as tofu boxes; falls back to matplotlib's default silently.
for _candidate in ["Noto Sans Devanagari", "FreeSans", "Lohit Devanagari", "Nirmala UI"]:
    _matches = [f.name for f in fm.fontManager.ttflist if _candidate.lower() in f.name.lower()]
    if _matches:
        matplotlib.rcParams["font.family"] = _matches[0]
        break
else:
    import glob as _glob
    _free_sans = _glob.glob("/usr/share/fonts/**/FreeSans.ttf", recursive=True)
    if _free_sans:
        fm.fontManager.addfont(_free_sans[0])
        matplotlib.rcParams["font.family"] = fm.FontProperties(fname=_free_sans[0]).get_name()

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs import load_config
from model import build_model
from train.checkpoint import load_model_only


def _attn_forward_with_capture(self, x):
    """Monkey-patched forward that stashes softmax attention probs on the module."""
    B, T, D = x.shape
    q, k, v = self.qkv(x).split(D, dim=2)
    q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
    k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
    v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
    att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.d_head))
    att = att.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float("-inf"))
    att = torch.softmax(att, dim=-1)
    self.last_attn = att.detach()          # (B, n_heads, T, T) - kept for analysis
    att = self.attn_drop(att)
    y = att @ v
    y = y.transpose(1, 2).contiguous().view(B, T, D)
    return self.resid_drop(self.out_proj(y))


def enable_attention_capture(model):
    for block in model.blocks:
        block.attn.forward = _attn_forward_with_capture.__get__(block.attn, block.attn.__class__)


@torch.no_grad()
def get_attention_for_sentence(model, sp, sentence, device):
    """Run one forward pass and return (attn_per_layer, tokens)."""
    ids = sp.encode(sentence, out_type=int)[:model.max_seq_len]
    x = torch.tensor([ids], device=device)
    model.eval()
    model(x)
    attn_per_layer = [block.attn.last_attn[0].cpu().numpy() for block in model.blocks]  # each (n_heads,T,T)
    tokens = [sp.id_to_piece(i) for i in ids]
    return attn_per_layer, tokens


def entropy_and_distance(attn_per_layer):
    """Per (layer, head) attention entropy (nats) and mean attention distance (tokens).

    Matches the original notebook's algorithm exactly: for each query position q,
    only the causally-visible key positions [0..q] are considered, and that row is
    explicitly renormalized (guards against tiny float leakage into masked positions
    rather than relying on softmax(-inf) being exactly zero).
    """
    n_layers = len(attn_per_layer)
    n_heads = attn_per_layer[0].shape[0]
    entropy = np.zeros((n_layers, n_heads))
    distance = np.zeros((n_layers, n_heads))
    for l, attn in enumerate(attn_per_layer):
        T = attn.shape[-1]
        for h in range(n_heads):
            ents, dists = [], []
            for q_pos in range(T):
                row = attn[h, q_pos, :q_pos + 1]
                row = row / (row.sum() + 1e-12)
                ents.append(-np.sum(row * np.log(row + 1e-12)))
                key_positions = np.arange(q_pos + 1)
                dists.append(np.sum(row * np.abs(q_pos - key_positions)))
            entropy[l, h] = np.mean(ents)
            distance[l, h] = np.mean(dists)
    return entropy, distance


def plot_heatmaps(attn_per_layer, tokens, out_path, layer_indices=None, head_indices=None):
    """Plot query-vs-key heatmaps for the given layers/heads (default: first+last layer, heads 0-3)."""
    n_heads_available = attn_per_layer[0].shape[0]
    if layer_indices is None:
        layer_indices = [0, len(attn_per_layer) - 1]
    if head_indices is None:
        head_indices = list(range(min(4, n_heads_available)))
    else:
        head_indices = [h for h in head_indices if h < n_heads_available]
    if not head_indices:
        raise ValueError(f"No valid heads to plot (model only has {n_heads_available} head(s))")

    n_rows, n_cols = len(layer_indices), len(head_indices)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 6 * n_rows))
    axes = np.atleast_2d(axes)

    for r, layer_idx in enumerate(layer_indices):
        for c, head_idx in enumerate(head_indices):
            ax = axes[r, c]
            attn = attn_per_layer[layer_idx][head_idx]
            ax.imshow(attn, cmap="viridis", aspect="auto")
            ax.set_title(f"Layer {layer_idx} Head {head_idx}")
            ax.set_xticks(range(len(tokens)))
            ax.set_xticklabels(tokens, rotation=90, fontsize=6)
            ax.set_yticks(range(len(tokens)))
            ax.set_yticklabels(tokens, fontsize=6)
            ax.set_xlabel("Key position")
            ax.set_ylabel("Query position")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved heatmap grid -> {out_path}")


def plot_entropy_distance(entropy, distance, out_prefix):
    fig, ax = plt.subplots(figsize=(6.5, 5))
    im = ax.imshow(entropy, cmap="magma", aspect="auto")
    ax.set_xlabel("Head"); ax.set_ylabel("Layer")
    ax.set_title("Attention entropy per layer/head")
    plt.colorbar(im, ax=ax, label="entropy (nats)")
    plt.tight_layout(); plt.savefig(f"{out_prefix}_entropy.png", dpi=150); plt.close()

    fig, ax = plt.subplots(figsize=(6.5, 5))
    im = ax.imshow(distance, cmap="coolwarm", aspect="auto")
    ax.set_xlabel("Head"); ax.set_ylabel("Layer")
    ax.set_title("Mean attention distance per layer/head")
    plt.colorbar(im, ax=ax, label="mean distance (tokens)")
    plt.tight_layout(); plt.savefig(f"{out_prefix}_distance.png", dpi=150); plt.close()

    print(f"Saved {out_prefix}_entropy.png and {out_prefix}_distance.png")
    print("Attention entropy (rows=layers, cols=heads, nats):\n", np.round(entropy, 3))
    print("Mean attention distance (rows=layers, cols=heads, tokens):\n", np.round(distance, 2))


def main(config_path, checkpoint_path, sentence, out_path):
    cfg = load_config(config_path)
    model = build_model(cfg).to(cfg["device"])
    load_model_only(checkpoint_path, model, cfg["device"])
    model.eval()
    enable_attention_capture(model)

    sp = spm.SentencePieceProcessor(model_file=cfg["sp_model"])

    if sentence is None:
        # default: first 60 tokens of the held-out validation file
        with open(cfg["val_txt"], encoding="utf-8") as f:
            text = f.readline().strip()
        ids = sp.encode(text, out_type=int)[:60]
        sentence = sp.decode(ids)

    attn_per_layer, tokens = get_attention_for_sentence(model, sp, sentence, cfg["device"])
    print(f"layers: {len(attn_per_layer)} | shape per layer: {attn_per_layer[0].shape}")
    print("tokens:", tokens)

    plot_heatmaps(attn_per_layer, tokens, out_path)

    entropy, distance = entropy_and_distance(attn_per_layer)
    out_prefix = os.path.splitext(out_path)[0]
    plot_entropy_distance(entropy, distance, out_prefix)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--sentence", default=None, help="Optional sentence to visualize; "
                                                           "defaults to the first held-out validation line")
    parser.add_argument("--out", default="attention_heatmap.png")
    args = parser.parse_args()
    main(args.config, args.checkpoint, args.sentence, args.out)
