"""
Download Sangraha (Hindi, verified split) up to a token budget,
then merge with your cleaned crawl corpus into one final .txt
ready for tokenizer training.

pip install datasets --break-system-packages
"""

from datasets import load_dataset
import hashlib

TARGET_TOKENS = 400_000_000

CRAWL_CLEAN_FILE = r"C:\Users\satya\Desktop\LMA_PROJ\Hindi\hindi_corpus_clean.txt"
PUBLIC_OUT_FILE  = r"C:\Users\satya\Desktop\LMA_PROJ\Hindi\sangraha_hin_verified.txt"
FINAL_OUT_FILE   = r"C:\Users\satya\Desktop\LMA_PROJ\Hindi\final_corpus.txt"

# crude tokens-per-word estimate for Hindi (Devanagari tokenizers run ~1.7x words)
TOKENS_PER_WORD = 1.7


def download_sangraha(target_tokens=TARGET_TOKENS, out_file=PUBLIC_OUT_FILE):
    ds = load_dataset(
        "ai4bharat/sangraha",
        data_dir="verified/hin",
        split="train",
        streaming=True,
    )
    words_written = 0
    target_words = int(target_tokens / TOKENS_PER_WORD)

    with open(out_file, "w", encoding="utf-8") as f:
        for row in ds:
            text = row.get("text", "").strip()
            if not text:
                continue
            text = " ".join(text.split())  # flatten to one line, one doc/line
            f.write(text + "\n")
            words_written += len(text.split())
            if words_written >= target_words:
                break

    print(f"[sangraha] wrote ~{words_written:,} words (~{int(words_written*TOKENS_PER_WORD):,} tokens) -> {out_file}")


def merge_with_dedup(files, final_out=FINAL_OUT_FILE):
    """Concat + drop exact-duplicate lines across sources (crawl vs public
    may share the same news article)."""
    seen = set()
    kept = 0
    with open(final_out, "w", encoding="utf-8") as out_f:
        for path in files:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    h = hashlib.sha256(line.encode("utf-8")).hexdigest()
                    if h in seen:
                        continue
                    seen.add(h)
                    out_f.write(line + "\n")
                    kept += 1
    print(f"[merge] {kept:,} unique lines -> {final_out}")


if __name__ == "__main__":
    download_sangraha()
    merge_with_dedup([PUBLIC_OUT_FILE, CRAWL_CLEAN_FILE])
