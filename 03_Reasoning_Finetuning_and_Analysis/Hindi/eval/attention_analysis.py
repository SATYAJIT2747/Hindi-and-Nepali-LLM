import os
import sys
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sentencepiece as spm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# Ensure UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# Attempt to configure Devanagari font for matplotlib
for _candidate in ["Noto Sans Devanagari", "FreeSans", "Lohit Devanagari", "Nirmala UI", "Arial"]:
    _matches = [f.name for f in fm.fontManager.ttflist if _candidate.lower() in f.name.lower()]
    if _matches:
        matplotlib.rcParams["font.family"] = _matches[0]
        break

# Path Resolution
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUT_DIR = os.path.join(BASE_DIR, "attention_analysis")
os.makedirs(OUT_DIR, exist_ok=True)

TOKENIZER_PATH = os.path.join(DATA_DIR, "hindi_bpe_32000.model")
PRETRAINED_CKPT = os.path.join(DATA_DIR, "hindi_gpt.pt")
SOFT_CKPT = os.path.join(DATA_DIR, "soft_best.pt")
HARD_CKPT = os.path.join(DATA_DIR, "hard_best.pt")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_SEQ_LEN = 512

# Tokenizer
if not os.path.exists(TOKENIZER_PATH):
    raise FileNotFoundError(f"Tokenizer not found: {TOKENIZER_PATH}")
sp = spm.SentencePieceProcessor()
sp.load(TOKENIZER_PATH)
VOCAB_SIZE = sp.get_piece_size()

# =============================================================================
# MODEL DEFINITION & ATTENTION CAPTURE
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
        mask = torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1).bool()
        self.register_buffer("causal_mask", mask.view(1, 1, max_seq_len, max_seq_len), persistent=True)
        self.last_attn = None

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.nhead, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.nhead, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.nhead, self.head_dim).transpose(1, 2)

        attn_scores = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        causal_mask = self.causal_mask[:, :, :T, :T]
        attn_scores = attn_scores.masked_fill(causal_mask, float("-inf"))
        attn_probs = torch.softmax(attn_scores, dim=-1)
        self.last_attn = attn_probs.detach()  # (B, nhead, T, T)

        y = attn_probs @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(y)


class TransformerBlock(nn.Module):
    def __init__(self, d_model=448, nhead=8, dim_feedforward=1024, max_seq_len=512, dropout=0.1):
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
        x = x + self.dropout(self.ffn["fc2"](F.gelu(self.ffn["fc1"](self.ln2(x)))))
        return x


class HindiGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.vocab_size = VOCAB_SIZE
        self.d_model = 448
        self.max_seq_len = MAX_SEQ_LEN
        self.tok_emb = nn.Embedding(self.vocab_size, self.d_model)
        self.pos_emb = nn.Embedding(self.max_seq_len, self.d_model)
        self.drop = nn.Dropout(0.1)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model=448, nhead=8, dim_feedforward=1024, max_seq_len=MAX_SEQ_LEN, dropout=0.1)
            for _ in range(6)
        ])
        self.ln_f = nn.LayerNorm(self.d_model)
        self.lm_head = nn.Linear(self.d_model, self.vocab_size, bias=False)

    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device).unsqueeze(0)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.lm_head(x)


def load_model(ckpt_path):
    model = HindiGPT().to(DEVICE)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint missing: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    # Strip causal_mask buffers if incompatible shape
    state_dict = {k: v for k, v in state_dict.items() if not k.endswith("causal_mask")}
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


@torch.no_grad()
def extract_attention(model, prompt_text):
    ids = sp.encode(prompt_text, out_type=int)[:MAX_SEQ_LEN]
    x = torch.tensor([ids], device=DEVICE)
    model(x)
    attn_per_layer = [block.attn.last_attn[0].cpu().numpy() for block in model.blocks]  # 6 layers x (8, T, T)
    tokens = [sp.id_to_piece(i) for i in ids]
    return attn_per_layer, tokens


def compute_layer_metrics(attn_layer):
    """attn_layer: (nhead, T, T)"""
    nhead, T, _ = attn_layer.shape
    entropies = []
    distances = []
    concentrations = []

    for h in range(nhead):
        h_ents, h_dists, h_concs = [], [], []
        for q in range(T):
            row = attn_layer[h, q, :q+1]
            row = row / (row.sum() + 1e-12)
            ent = -np.sum(row * np.log(row + 1e-12))
            pos_dist = np.sum(row * np.abs(q - np.arange(q + 1)))
            conc = np.max(row)

            h_ents.append(ent)
            h_dists.append(pos_dist)
            h_concs.append(conc)

        entropies.append(np.mean(h_ents))
        distances.append(np.mean(h_dists))
        concentrations.append(np.mean(h_concs))

    return np.mean(entropies), np.mean(distances), np.mean(concentrations)


# Sample Prompts in Hindi
PROMPTS = {
    "compare_2way": "दीपक के अंक = 70\nसुरेश के अंक = 77\n\nसबसे अधिक अंक किसके हैं?\nउत्तर:",
    "compare_3way": "संजय के पास मूल्य 35 रुपये हैं।\nराहुल के पास मूल्य 37 रुपये हैं।\nसुरेश के पास मूल्य 32 रुपये हैं।\n\nसबसे अधिक मूल्य किसके हैं?\nउत्तर:",
    "compare_4way": "अमित: 57 वर्ष\nदीपक: 53 वर्ष\nसंजय: 59 वर्ष\nअशोक: 51 वर्ष\n\nसबसे छोटे\nउत्तर:"
}


def plot_layer_comparison(attn_dict, tokens, layer_idx, task_name):
    """attn_dict: {'Pretrained': attn_pre, 'Soft': attn_soft, 'Hard': attn_hard}"""
    models = ["Pretrained", "Soft", "Hard"]
    n_heads = 8
    fig, axes = plt.subplots(len(models), n_heads, figsize=(24, 9))

    for r, m_name in enumerate(models):
        attn_layer = attn_dict[m_name][layer_idx]  # (8, T, T)
        for h in range(n_heads):
            ax = axes[r, h]
            im = ax.imshow(attn_layer[h], cmap="viridis", aspect="auto")
            if r == 0:
                ax.set_title(f"Head {h}", fontsize=11, fontweight="bold")
            if h == 0:
                ax.set_ylabel(f"{m_name}\n(Query Pos)", fontsize=10, fontweight="bold")
            ax.set_xticks([])
            ax.set_yticks([])

    plt.suptitle(f"{task_name} — Layer {layer_idx} Attention Comparison across Models", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_img = os.path.join(OUT_DIR, f"{task_name}_layer{layer_idx}_comparison.png")
    plt.savefig(out_img, dpi=150)
    plt.close()
    print(f"Saved: {out_img}")


def plot_metrics(summary_metrics, task_name):
    """Plot entropy, distance, and concentration for a task"""
    models = ["Pretrained", "Soft", "Hard"]
    layers = [0, 5]

    # Metrics
    for metric_name in ["entropy", "distance", "concentration"]:
        fig, ax = plt.subplots(figsize=(8, 5))
        bar_width = 0.25
        x = np.arange(len(layers))

        for i, m_name in enumerate(models):
            vals = [summary_metrics[task_name][m_name][f"L{l}"][metric_name] for l in layers]
            ax.bar(x + i * bar_width, vals, width=bar_width, label=m_name)

        ax.set_xticks(x + bar_width)
        ax.set_xticklabels(["Layer 0 (Early)", "Layer 5 (Late)"])
        ax.set_ylabel(metric_name.capitalize())
        ax.set_title(f"{task_name} — Attention {metric_name.capitalize()} (Early vs Late Layer)")
        ax.legend()
        plt.tight_layout()
        out_path = os.path.join(OUT_DIR, f"{task_name}_metrics_{metric_name}.png")
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"Saved: {out_path}")


def main():
    print("================================================================================")
    print("POST-FINETUNING ATTENTION ANALYSIS (HINDI)")
    print("================================================================================")

    print("Loading models...")
    models = {
        "Pretrained": load_model(PRETRAINED_CKPT),
        "Soft": load_model(SOFT_CKPT),
        "Hard": load_model(HARD_CKPT)
    }
    print("[OK] All 3 checkpoints loaded successfully.\n")

    summary_file = os.path.join(OUT_DIR, "analysis_summary.txt")
    summary_metrics = {}

    with open(summary_file, "w", encoding="utf-8") as f_sum:
        f_sum.write("ATTENTION ANALYSIS SUMMARY (HINDI)\n")
        f_sum.write("================================================================================\n\n")

        for task_name, prompt_text in PROMPTS.items():
            print(f"Analyzing {task_name}...")
            f_sum.write(f"{task_name}\n")
            f_sum.write("----------------------------------------\n")

            attn_dict = {}
            summary_metrics[task_name] = {}

            for m_name, model in models.items():
                attn_per_layer, tokens = extract_attention(model, prompt_text)
                attn_dict[m_name] = attn_per_layer
                summary_metrics[task_name][m_name] = {}

                for l in [0, 5]:
                    ent, dist, conc = compute_layer_metrics(attn_per_layer[l])
                    summary_metrics[task_name][m_name][f"L{l}"] = {
                        "entropy": ent,
                        "distance": dist,
                        "concentration": conc
                    }

            # Write Entropy
            f_sum.write("ENTROPY (Lower = More Focused)\n")
            for m in ["Pretrained", "Soft", "Hard"]:
                e0 = summary_metrics[task_name][m]["L0"]["entropy"]
                e5 = summary_metrics[task_name][m]["L5"]["entropy"]
                f_sum.write(f"  {m:<12} | L0: {e0:.4f} | L5: {e5:.4f}\n")
            f_sum.write("\n")

            # Write Distance
            f_sum.write("DISTANCE (Lower = More Local)\n")
            for m in ["Pretrained", "Soft", "Hard"]:
                d0 = summary_metrics[task_name][m]["L0"]["distance"]
                d5 = summary_metrics[task_name][m]["L5"]["distance"]
                f_sum.write(f"  {m:<12} | L0: {d0:.4f} | L5: {d5:.4f}\n")
            f_sum.write("\n")

            # Write Concentration
            f_sum.write("CONCENTRATION (Higher = More Peaked)\n")
            for m in ["Pretrained", "Soft", "Hard"]:
                c0 = summary_metrics[task_name][m]["L0"]["concentration"]
                c5 = summary_metrics[task_name][m]["L5"]["concentration"]
                f_sum.write(f"  {m:<12} | L0: {c0:.4f} | L5: {c5:.4f}\n")
            f_sum.write("\n\n")

            # Generate Plots
            plot_layer_comparison(attn_dict, tokens, 0, task_name)
            plot_layer_comparison(attn_dict, tokens, 5, task_name)
            plot_metrics(summary_metrics, task_name)

    print("\n================================================================================")
    print(f"Analysis Complete! Summary saved to {summary_file}")
    print("================================================================================")


if __name__ == "__main__":
    main()
