import os
import sys
import numpy as np
import torch
import sentencepiece as spm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from bonus.config import SP_MODEL_PATH, VAL_TXT_PATH, CKPT_DIR, MODEL_CONFIG, WORKDIR
from bonus.model import GPT

def get_devanagari_font():
    possible_fonts = [
        "/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansDevanagari-Regular.ttf",
    ]
    for font_path in possible_fonts:
        if os.path.exists(font_path):
            return font_manager.FontProperties(fname=font_path)
    for font_path in font_manager.findSystemFonts():
        if "NotoSansDevanagari" in font_path or "Nirmala" in font_path:
            return font_manager.FontProperties(fname=font_path)
    return None

def get_attention_and_stats(model, sp, sentence, device):
    ids = sp.encode(sentence, out_type=int)[:60]
    if len(ids) == 0:
        raise ValueError("Sentence produced zero tokens.")

    x = torch.tensor([ids], dtype=torch.long, device=device)
    model.eval()
    with torch.no_grad():
        model(x)

    attn_per_layer = [block.attn.last_attn[0].detach().cpu().numpy() for block in model.blocks]
    tokens = [sp.id_to_piece(i) for i in ids]
    n_layers = len(attn_per_layer)
    n_heads = attn_per_layer[0].shape[0]

    entropy = np.zeros((n_layers, n_heads), dtype=np.float64)
    distance = np.zeros((n_layers, n_heads), dtype=np.float64)

    for l, attn in enumerate(attn_per_layer):
        T = attn.shape[-1]
        for h in range(n_heads):
            ents, dists = [], []
            for q_pos in range(T):
                row = attn[h, q_pos, :q_pos + 1]
                row = row / (row.sum() + 1e-12)
                ent = -np.sum(row * np.log(row + 1e-12))
                key_positions = np.arange(q_pos + 1)
                dist = np.sum(row * np.abs(q_pos - key_positions))
                ents.append(ent)
                dists.append(dist)
            entropy[l, h] = np.mean(ents)
            distance[l, h] = np.mean(dists)

    return attn_per_layer, tokens, entropy, distance

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sp = spm.SentencePieceProcessor(model_file=SP_MODEL_PATH)
    font_prop = get_devanagari_font()

    model = GPT(**MODEL_CONFIG).to(device)
    best_path = os.path.join(CKPT_DIR, "best.pt")
    if os.path.exists(best_path):
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
        print(f"Loaded checkpoint from {best_path}")
    model.eval()

    if os.path.exists(VAL_TXT_PATH):
        with open(VAL_TXT_PATH, encoding="utf-8") as f:
            sample_sentence = f.readline().strip()
    else:
        sample_sentence = "भारत एक विशाल और विविधताओं से भरा देश है।"

    attn_per_layer, tokens, entropy, distance = get_attention_and_stats(model, sp, sample_sentence, device)
    n_layers, n_heads = len(attn_per_layer), attn_per_layer[0].shape[0]

    graphs_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "graphs"))
    os.makedirs(graphs_dir, exist_ok=True)

    # Plot 1: Individual layer plots
    tick_step = 5
    T = len(tokens)
    tick_positions = list(range(0, T, tick_step))
    tick_labels = [tokens[i] for i in tick_positions]

    for layer_idx in range(n_layers):
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()
        for head_idx in range(n_heads):
            ax = axes[head_idx]
            attn = attn_per_layer[layer_idx][head_idx]
            im = ax.imshow(attn, cmap="viridis", aspect="equal", interpolation="nearest")
            ax.set_title(f"Layer {layer_idx} | Head {head_idx}\nEntropy={entropy[layer_idx, head_idx]:.3f} | Dist={distance[layer_idx, head_idx]:.2f}")
            ax.set_xticks(tick_positions)
            ax.set_yticks(tick_positions)
            if font_prop:
                ax.set_xticklabels(tick_labels, rotation=90, fontsize=8, fontproperties=font_prop)
                ax.set_yticklabels(tick_labels, fontsize=8, fontproperties=font_prop)
            else:
                ax.set_xticklabels(tick_labels, rotation=90, fontsize=8)
                ax.set_yticklabels(tick_labels, fontsize=8)
            ax.set_xlabel("Key Position")
            ax.set_ylabel("Query Position")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        fig.suptitle(f"Ablated Hindi GPT — Layer {layer_idx} Heatmaps (No Positional Embeddings)", fontsize=18, y=1.02)
        plt.tight_layout()
        out_file = os.path.join(graphs_dir, f"layer_{layer_idx}_attention.png")
        plt.savefig(out_file, dpi=180, bbox_inches="tight")
        plt.close(fig)

    # Plot 2: Combined 48-head heatmap
    fig, axes = plt.subplots(n_layers, n_heads, figsize=(24, 30))
    for layer_idx in range(n_layers):
        for head_idx in range(n_heads):
            ax = axes[layer_idx, head_idx]
            attn = attn_per_layer[layer_idx][head_idx]
            ax.imshow(attn, cmap="viridis", aspect="equal", interpolation="nearest")
            ax.set_title(f"L{layer_idx} H{head_idx}\nH={entropy[layer_idx, head_idx]:.2f}, D={distance[layer_idx, head_idx]:.1f}", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
            if head_idx == 0:
                ax.set_ylabel(f"Layer {layer_idx}", fontsize=11)

    fig.suptitle("Ablated Hindi GPT — All 48 Attention Heads (No Positional Embeddings)", fontsize=20, y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.985])
    combined_path = os.path.join(graphs_dir, "all_48_attention_heatmaps.png")
    plt.savefig(combined_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved all attention heatmaps and 48-head plot to {graphs_dir}")

if __name__ == "__main__":
    main()
