import os
import sys
import random
from collections import Counter
import numpy as np
import torch
import torch.nn.functional as F
import sentencepiece as spm
import sacrebleu
from rouge_score import rouge_scorer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from bonus.config import SP_MODEL_PATH, VAL_TXT_PATH, CKPT_DIR, MODEL_CONFIG
from bonus.model import GPT

_rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)

def repetition_rate(text, n=4):
    tokens = text.split()
    if len(tokens) < n:
        return 0.0
    ngrams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    counts = Counter(ngrams)
    repeated = sum(c for c in counts.values() if c > 1)
    return repeated / len(ngrams)

def distinct_n(texts, n=1):
    all_ngrams = []
    for t in texts:
        toks = t.split()
        all_ngrams.extend([tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)])
    return len(set(all_ngrams)) / len(all_ngrams) if all_ngrams else 0.0

def compute_metrics(hyps, refs):
    bleu = sacrebleu.corpus_bleu(hyps, [refs])
    chrf = sacrebleu.corpus_chrf(hyps, [refs])
    chrfpp = sacrebleu.corpus_chrf(hyps, [refs], word_order=2)
    rouge_l = np.mean([_rouge.score(r, h)["rougeL"].fmeasure for h, r in zip(hyps, refs)])
    rep_rate = np.mean([repetition_rate(h) for h in hyps])
    d1, d2 = distinct_n(hyps, 1), distinct_n(hyps, 2)
    return dict(
        BLEU4=bleu.score,
        chrF=chrf.score,
        chrFpp=chrfpp.score,
        ROUGE_L=rouge_l * 100,
        repetition_rate=rep_rate,
        distinct1=d1,
        distinct2=d2
    )

@torch.no_grad()
def generate(model, prefix_ids, max_new_tokens=40, temperature=1.0, greedy=False, device="cuda"):
    idx = torch.tensor([prefix_ids], dtype=torch.long, device=device)
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.max_seq_len:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :]
        if greedy:
            next_id = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            probs = F.softmax(logits / temperature, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_id], dim=1)
    return idx[0].tolist()

@torch.no_grad()
def generate_topk_topp(model, prefix_ids, max_new_tokens=40, temperature=1.0, top_k=None, top_p=None, device="cuda"):
    idx = torch.tensor([prefix_ids], dtype=torch.long, device=device)
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.max_seq_len:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / temperature
        if top_k is not None:
            k = min(top_k, logits.size(-1))
            values, _ = torch.topk(logits, k)
            logits = torch.where(logits < values[:, -1].unsqueeze(-1), torch.full_like(logits, float("-inf")), logits)
        if top_p is not None:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            sorted_probs = F.softmax(sorted_logits, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
            sorted_indices_to_remove[:, 0] = False
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits = logits.masked_fill(indices_to_remove, float("-inf"))
        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_id], dim=1)
    return idx[0].tolist()

@torch.no_grad()
def generate_beam_search(model, prefix_ids, max_new_tokens=40, num_beams=4, device="cuda"):
    beams = [(prefix_ids[:], 0.0)]
    for _ in range(max_new_tokens):
        candidates = []
        for seq, score in beams:
            idx = torch.tensor([seq], dtype=torch.long, device=device)[:, -model.max_seq_len:]
            logits, _ = model(idx)
            log_probs = F.log_softmax(logits[:, -1, :], dim=-1)
            topk_logp, topk_idx = torch.topk(log_probs, num_beams, dim=-1)
            for lp, tok in zip(topk_logp[0].tolist(), topk_idx[0].tolist()):
                candidates.append((seq + [tok], score + lp))
        candidates.sort(key=lambda c: c[1], reverse=True)
        beams = candidates[:num_beams]
    best_seq, _ = max(beams, key=lambda c: c[1])
    return best_seq

def build_holdout_examples(val_txt_path, sp, n_examples=50, prefix_tokens=20, cont_tokens=40, seed=0):
    random.seed(seed)
    lines = [l.strip() for l in open(val_txt_path, encoding="utf-8") if l.strip()]
    examples = []
    for line in lines:
        ids = sp.encode(line, out_type=int)
        if len(ids) >= prefix_tokens + cont_tokens:
            examples.append(ids)
    random.shuffle(examples)
    examples = examples[:n_examples]
    data = []
    for ids in examples:
        prefix_ids, ref_ids = ids[:prefix_tokens], ids[prefix_tokens:prefix_tokens + cont_tokens]
        data.append({
            "prefix_ids": prefix_ids,
            "ref_ids": ref_ids,
            "prefix_text": sp.decode(prefix_ids),
            "ref_text": sp.decode(ref_ids),
        })
    return data

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sp = spm.SentencePieceProcessor(model_file=SP_MODEL_PATH)

    print("Loading Ablated Model H (No Positional Embeddings)...")
    model = GPT(**MODEL_CONFIG).to(device)
    best_path = os.path.join(CKPT_DIR, "best.pt")
    if os.path.exists(best_path):
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
        print(f"Loaded checkpoint from {best_path}")
    else:
        print(f"⚠️ Checkpoint {best_path} not found. Running evaluation on randomly initialized model.")
    model.eval()

    holdout = build_holdout_examples(VAL_TXT_PATH, sp, n_examples=50)

    settings = {
        "greedy":     lambda m, p: generate(m, p, 40, greedy=True, device=device),
        "temp_0.5":   lambda m, p: generate(m, p, 40, temperature=0.5, device=device),
        "temp_1.0":   lambda m, p: generate(m, p, 40, temperature=1.0, device=device),
        "temp_1.5":   lambda m, p: generate(m, p, 40, temperature=1.5, device=device),
        "top_k_20":   lambda m, p: generate_topk_topp(m, p, 40, top_k=20, device=device),
        "top_k_50":   lambda m, p: generate_topk_topp(m, p, 40, top_k=50, device=device),
        "top_p_0.8":  lambda m, p: generate_topk_topp(m, p, 40, top_p=0.8, device=device),
        "top_p_0.9":  lambda m, p: generate_topk_topp(m, p, 40, top_p=0.9, device=device),
        "top_p_0.95": lambda m, p: generate_topk_topp(m, p, 40, top_p=0.95, device=device),
        "beam_4":     lambda m, p: generate_beam_search(m, p, 40, num_beams=4, device=device),
    }

    print("\n=== GENERATION EVALUATION: ABLATED MODEL H (NO POSITIONAL EMBEDDINGS) ===")
    for name, decode_fn in settings.items():
        hyps, refs = [], []
        for ex in holdout:
            gen_ids = decode_fn(model, ex["prefix_ids"])[len(ex["prefix_ids"]):]
            hyps.append(sp.decode(gen_ids))
            refs.append(ex["ref_text"])
        m = compute_metrics(hyps, refs)
        print(f"{name:12s} | BLEU4: {m['BLEU4']:6.3f} | chrF: {m['chrF']:6.3f} | ROUGE-L: {m['ROUGE_L']:6.3f} | RepRate: {m['repetition_rate']:6.4f} | Dist-1: {m['distinct1']:6.4f} | Dist-2: {m['distinct2']:6.4f}")

if __name__ == "__main__":
    main()
