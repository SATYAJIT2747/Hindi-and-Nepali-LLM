"""BLEU / chrF / chrF++ / ROUGE-L plus fluency/diversity diagnostics
(repetition rate, Distinct-1, Distinct-2)."""
from collections import Counter

import numpy as np
import sacrebleu
from rouge_score import rouge_scorer

_rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)


def repetition_rate(text, n=4):
    """Fraction of n-grams in `text` that are repeats of an earlier n-gram."""
    tokens = text.split()
    if len(tokens) < n:
        return 0.0
    ngrams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    counts = Counter(ngrams)
    repeated = sum(c for c in counts.values() if c > 1)
    return repeated / len(ngrams)


def distinct_n(texts, n=1):
    """Distinct-n: unique n-grams / total n-grams over a list of generated texts."""
    all_ngrams = []
    for t in texts:
        toks = t.split()
        all_ngrams.extend([tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)])
    return len(set(all_ngrams)) / len(all_ngrams) if all_ngrams else 0.0


def compute_metrics(hyps, refs):
    """Compute the full metric suite for one decoding setting."""
    bleu = sacrebleu.corpus_bleu(hyps, [refs])                # BLEU-4 by default
    chrf = sacrebleu.corpus_chrf(hyps, [refs])                # chrF
    chrfpp = sacrebleu.corpus_chrf(hyps, [refs], word_order=2)  # chrF++
    rouge_l = np.mean([_rouge.score(r, h)["rougeL"].fmeasure for h, r in zip(hyps, refs)])
    rep_rate = np.mean([repetition_rate(h) for h in hyps])
    d1, d2 = distinct_n(hyps, 1), distinct_n(hyps, 2)
    return dict(
        BLEU4=bleu.score, chrF=chrf.score, chrFpp=chrfpp.score,
        ROUGE_L=rouge_l * 100, repetition_rate=rep_rate,
        distinct1=d1, distinct2=d2,
    )


def print_metrics(name, metrics):
    print(f"\n=== {name} ===")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
