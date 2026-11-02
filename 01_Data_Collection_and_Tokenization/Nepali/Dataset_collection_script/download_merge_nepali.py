"""
1) Downloads ai4bharat/sangraha verified/nep parquet files directly (no `datasets` lib needed).
2) Converts them to flat one-doc-per-line .txt (same shape as your crawl output).
3) Merges with your cleaned Nepali corpus, deduping exact-match lines.

pip install huggingface_hub pyarrow --break-system-packages
"""

from huggingface_hub import list_repo_files, hf_hub_download
import pyarrow.parquet as pq
import hashlib
import os

REPO = "ai4bharat/sangraha"
SUBFOLDER = "verified/nep"
LOCAL_PARQUET_DIR = r"C:\Users\satya\Desktop\LMA_PROJ\Nepali\sangraha_nep_raw"
SANGRAHA_TXT_OUT = r"C:\Users\satya\Desktop\LMA_PROJ\Nepali\sangraha_nep.txt"
NEPALI_CLEAN_FILE = r"C:\Users\satya\Desktop\LMA_PROJ\Nepali\nepali_corpus_clean.txt"
FINAL_OUT_FILE = r"C:\Users\satya\Desktop\LMA_PROJ\Nepali\final_nepali_corpus.txt"

TARGET_TOKENS = 400_000_000
TOKENS_PER_WORD = 1.7  # rough Devanagari estimate


def download_and_process(out_file=SANGRAHA_TXT_OUT, target_tokens=TARGET_TOKENS):
    os.makedirs(LOCAL_PARQUET_DIR, exist_ok=True)
    files = [f for f in list_repo_files(REPO, repo_type="dataset")
             if f.startswith(SUBFOLDER) and f.endswith(".parquet")]
    files.sort()
    print(f"[info] found {len(files)} parquet files in {SUBFOLDER}")

    target_words = int(target_tokens / TOKENS_PER_WORD)
    words_written = 0

    with open(out_file, "w", encoding="utf-8") as out_f:
        for f in files:
            path = hf_hub_download(repo_id=REPO, repo_type="dataset", filename=f,
                                    local_dir=LOCAL_PARQUET_DIR)
            print(f"[downloaded] {f}")

            pf = pq.ParquetFile(path)
            for batch in pf.iter_batches(batch_size=1000, columns=["text"]):
                for text in batch.column("text").to_pylist():
                    if not text:
                        continue
                    flat = " ".join(text.split())
                    out_f.write(flat + "\n")
                    words_written += len(flat.split())
                if words_written >= target_words:
                    print(f"[stop] hit target ~{words_written:,} words, stopping after {f}")
                    return
            print(f"[processed] {os.path.basename(path)} -> {words_written:,} words so far")

            # delete parquet immediately to save disk space
            try:
                os.remove(path)
            except OSError:
                pass


def merge_with_dedup(files, final_out=FINAL_OUT_FILE):
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
    download_and_process()
    merge_with_dedup([SANGRAHA_TXT_OUT, NEPALI_CLEAN_FILE])