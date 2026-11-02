"""
Deliverable #5: Tokenizer training code (Hindi)
Also produces deliverables #6 (vocabulary file) and #7 (tokenizer model file).

  1. Splits corpus into TRAIN / VALIDATION / TEST (proper 3-way split, seeded):
       - train: fits the tokenizer (BPE merges learned from this data only)
       - validation: used to CHOOSE the vocab size (hyperparameter selection)
       - test: held out and touched only ONCE, after vocab size is already
         chosen, to produce the final unbiased reported numbers
  2. Trains SentencePiece BPE tokenizers from scratch at multiple vocab sizes
  3. Evaluates each candidate on VALIDATION: fertility, UNK rate, chars/token
  4. Picks the vocab size with the best fertility/UNK trade-off on validation:
     smallest vocab within 3% fertility of the best achievable fertility
  5. Re-evaluates ONLY the chosen model on TEST for the final report numbers
     (test set never influenced which vocab size was picked)
  6. Copies the chosen model+vocab to final/ as the deliverable files
  7. Prints two separate reports: validation sweep (selection) and
     final test-set report (unbiased numbers: token-frequency stats,
     avg chars/token, tokenization examples, UNK statistics)

No pretrained tokenizer used. Trained from scratch on this corpus only.

pip install sentencepiece --break-system-packages
"""

import sentencepiece as spm
import os
import random
import shutil
from collections import Counter

# ---------------- CONFIG ----------------
LANG = "hindi"
CORPUS_PATH = r"C:\Users\satya\Desktop\LMA_PROJ\Hindi\data_merged_final\final_corpus.txt"
SAMPLE_WORDS = ["भारत", "विश्वविद्यालय", "सरकार", "अर्थव्यवस्था", "प्रधानमंत्री"]
VOCAB_SIZES = [16000, 32000, 50000, 64000]
VAL_FRACTION = 0.01          # used DURING training to choose vocab size (hyperparameter selection)
TEST_FRACTION = 0.01         # held out, touched only ONCE at the very end for final unbiased report
FERTILITY_TOLERANCE = 0.03   # pick smallest vocab within 3% of best fertility (on validation set)
WORK_DIR = "tokenizer_work_hindi"  # intermediate train/val/test splits + candidate models
FINAL_DIR = "final"                 # deliverable: final chosen model + vocab
SEED = 42
# -----------------------------------------


def split_train_val_test(src_path):
    """3-way split:
      train -> used to fit the tokenizer (SentencePiece BPE merges)
      val   -> used to CHOOSE the vocab size (hyperparameter selection)
      test  -> touched only once, after vocab size is already chosen, for the
               final reported numbers. Never used to make any decision.
    """
    train_path = os.path.join(WORK_DIR, f"{LANG}_train.txt")
    val_path = os.path.join(WORK_DIR, f"{LANG}_val.txt")
    test_path = os.path.join(WORK_DIR, f"{LANG}_test.txt")
    if os.path.exists(train_path) and os.path.exists(val_path) and os.path.exists(test_path):
        return train_path, val_path, test_path

    rng = random.Random(SEED)
    with open(src_path, encoding="utf-8") as f_in, \
         open(train_path, "w", encoding="utf-8") as f_train, \
         open(val_path, "w", encoding="utf-8") as f_val, \
         open(test_path, "w", encoding="utf-8") as f_test:
        for line in f_in:
            r = rng.random()
            if r < VAL_FRACTION:
                f_val.write(line)
            elif r < VAL_FRACTION + TEST_FRACTION:
                f_test.write(line)
            else:
                f_train.write(line)
    return train_path, val_path, test_path


def train_candidate(train_path, vocab_size):
    prefix = os.path.join(WORK_DIR, f"{LANG}_bpe_{vocab_size}")
    if os.path.exists(prefix + ".model"):
        return prefix
    spm.SentencePieceTrainer.train(
        input=train_path,
        model_prefix=prefix,
        vocab_size=vocab_size,
        model_type="bpe",
        character_coverage=0.9995,
        input_sentence_size=5_000_000,
        shuffle_input_sentence=True,
        num_threads=8,
        unk_id=0, bos_id=1, eos_id=2, pad_id=3,
    )
    return prefix


def evaluate(prefix, eval_path):
    sp = spm.SentencePieceProcessor(model_file=prefix + ".model")
    total_words, total_tokens, total_chars, unk_tokens = 0, 0, 0, 0
    token_counter = Counter()

    with open(eval_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ids = sp.encode(line, out_type=int)
            pieces = sp.encode(line, out_type=str)
            total_words += len(line.split())
            total_tokens += len(ids)
            total_chars += len(line)
            unk_tokens += ids.count(sp.unk_id())
            token_counter.update(pieces)

    return {
        "vocab_size": sp.get_piece_size(),
        "fertility": total_tokens / max(total_words, 1),
        "unk_rate": unk_tokens / max(total_tokens, 1),
        "avg_chars_per_token": total_chars / max(total_tokens, 1),
        "top_tokens_20": token_counter.most_common(20),
        "unique_tokens_seen": len(token_counter),
        "total_tokens": total_tokens,
        "unk_tokens": unk_tokens,
    }


def show_examples(prefix, sample_words):
    sp = spm.SentencePieceProcessor(model_file=prefix + ".model")
    print("  Tokenization examples:")
    for w in sample_words:
        print(f"    {w} -> {sp.encode(w, out_type=str)}")


def choose_best(results):
    """Smallest vocab size within FERTILITY_TOLERANCE of the best fertility."""
    best_fertility = min(m["fertility"] for _, m, _ in results)
    threshold = best_fertility * (1 + FERTILITY_TOLERANCE)
    candidates = [(vs, m, p) for vs, m, p in results if m["fertility"] <= threshold]
    return min(candidates, key=lambda x: x[0]), best_fertility


def main():
    os.makedirs(WORK_DIR, exist_ok=True)
    os.makedirs(FINAL_DIR, exist_ok=True)

    train_path, val_path, test_path = split_train_val_test(CORPUS_PATH)

    # ---- Step 1: train candidates, select vocab size using VALIDATION only ----
    results = []
    for vs in VOCAB_SIZES:
        print(f"[train] {LANG} vocab={vs} ...")
        prefix = train_candidate(train_path, vs)
        val_metrics = evaluate(prefix, val_path)
        results.append((vs, val_metrics, prefix))
        print(f"  [val] fertility={val_metrics['fertility']:.3f}  "
              f"unk_rate={val_metrics['unk_rate']:.5f}  "
              f"avg_chars/token={val_metrics['avg_chars_per_token']:.2f}")

    (chosen_vs, chosen_val_metrics, chosen_prefix), best_fertility = choose_best(results)

    print(f"\n[CHOSEN via validation set] {LANG}: vocab_size={chosen_vs}  "
          f"(val fertility={chosen_val_metrics['fertility']:.3f}, "
          f"best possible={best_fertility:.3f}, "
          f"within {FERTILITY_TOLERANCE*100:.0f}% tolerance)")

    # ---- Step 2: final unbiased report on TEST set (touched only now, once) ----
    test_metrics = evaluate(chosen_prefix, test_path)

    # ---- copy final deliverable files ----
    final_model = os.path.join(FINAL_DIR, f"{LANG}_tokenizer.model")
    final_vocab = os.path.join(FINAL_DIR, f"{LANG}_tokenizer.vocab")
    shutil.copy(chosen_prefix + ".model", final_model)
    shutil.copy(chosen_prefix + ".vocab", final_vocab)

    # ---- report (all final numbers are from the TEST set, never used for selection) ----
    print(f"\n===== FINAL REPORT ({LANG.capitalize()}) — computed on held-out TEST set =====")
    print(f"  Chosen vocab_size: {chosen_vs}  (selected using validation set above)")
    print(f"  Final model -> {final_model}")
    print(f"  Final vocab -> {final_vocab}")
    show_examples(chosen_prefix, SAMPLE_WORDS)

    print("\n  Token-frequency statistics (top 20, TEST set):")
    for tok, freq in test_metrics["top_tokens_20"]:
        print(f"    {tok!r:20s} freq={freq}")
    print(f"  Fertility (TEST):            {test_metrics['fertility']:.3f}")
    print(f"  Avg chars/token (TEST):      {test_metrics['avg_chars_per_token']:.2f}")
    print(f"  Unique tokens seen (TEST):   {test_metrics['unique_tokens_seen']}")
    print(f"  Total tokens (TEST):         {test_metrics['total_tokens']}")
    print(f"  UNK tokens (TEST):           {test_metrics['unk_tokens']}  "
          f"({test_metrics['unk_rate']*100:.3f}% of all tokens)")

    print(f"\n===== VALIDATION SWEEP SUMMARY ({LANG.capitalize()}) — used only to pick vocab size =====")
    print(f"{'vocab':>8} {'fertility':>10} {'unk_rate':>10} {'chars/tok':>10}")
    for vs, m, _ in results:
        marker = "  <== chosen" if vs == chosen_vs else ""
        print(f"{vs:>8} {m['fertility']:>10.3f} {m['unk_rate']:>10.5f} "
              f"{m['avg_chars_per_token']:>10.2f}{marker}")


if __name__ == "__main__":
    main()
