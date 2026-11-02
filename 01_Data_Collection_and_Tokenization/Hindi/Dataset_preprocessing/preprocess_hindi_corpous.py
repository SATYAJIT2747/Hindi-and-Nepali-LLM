"""
Corpus Preprocessor — JSONL shards -> single clean .txt for tokenizer input
=============================================================================

Takes the shard_*.jsonl files produced by the crawler and turns them into
ONE clean plaintext file, ready to concatenate with your public source
corpus before tokenizer training.

What it does:
  1. Loads every shard_*.jsonl in INPUT_DIR
  2. Per document:
       - Unicode NFC normalize
       - Strip known site boilerplate footer blocks (regex, editable below)
       - Line-by-line: drop lines that are mostly English/Latin script
         (ratio of Devanagari chars to Devanagari+Latin chars < threshold)
       - Drop lines that are pure noise (too short, no real content)
       - Flatten remaining lines into ONE line per document (tokenizer
         corpora conventionally want one doc/sentence per line, no
         embedded newlines)
  3. Deduplicates:
       - Exact-duplicate check on the cleaned text (hash of full text)
       - Near-duplicate check re-using the crawler's own signature
         (first 200 + last 200 chars + length band) in case the same
         article was crawled twice across different runs/resumes
  4. Drops docs that end up too short after cleaning (mostly-English
     pages, or pages that were almost entirely boilerplate)
  5. Writes ONE cleaned document per line to OUTPUT_FILE
  6. Prints a summary: docs in -> docs out, words in -> words out,
     how many were dropped and why

RUN:
    uv run python preprocess_corpus.py
"""

import glob
import hashlib
import json
import os
import re
import unicodedata

# ----------------------------------------------------------------------
# CONFIG — adjust these to your actual folders
# ----------------------------------------------------------------------
INPUT_DIR = r"C:\Users\satya\Desktop\LMA_PROJ\Hindi\hindi_corpus"
OUTPUT_FILE = r"C:\Users\satya\Desktop\LMA_PROJ\Hindi\hindi_corpus_clean.txt"

MIN_LINE_DEVANAGARI_RATIO = 0.5   # line kept only if >=50% of its
                                   # Devanagari+Latin letters are Devanagari
MIN_LINE_CHARS = 8                # drop very short/noise lines
MIN_DOC_WORDS_AFTER_CLEAN = 40    # drop whole doc if too little survives

DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
LATIN_LETTER_RE = re.compile(r"[A-Za-z]")
WORD_RE = re.compile(r"\S+")

# Known repeated site-footer / boilerplate blocks to strip outright
# before line filtering even runs. Add more patterns here as you spot
# them in your own data (one regex per recurring boilerplate block).
BOILERPLATE_PATTERNS = [
    re.compile(r"प्रभात खबर डिजिटल टॉप स्टोरी.*", re.DOTALL),
    re.compile(r"Prabhat Khabar App\s*:.*", re.DOTALL),
    re.compile(r"हम डाटा संग्रह टूल्स.*?Agree पर क्लिक करें।", re.DOTALL),  # cookie consent
    re.compile(r"\s*\|\s*Prabhat Khabar Network\s*"),
]

# Lines that are internal nav clutter, not article content -> dropped
# outright regardless of their Devanagari ratio (they're fluent Hindi,
# so the language filter alone won't catch them).
NOISE_LINE_PATTERNS = [
    re.compile(r"^यह भी पढ़ें"),
    re.compile(r"^Also Read"),
    re.compile(r"^और पढ़ें$"),
    re.compile(r"^\s*-\s*कोई Open IPOs डेटा उपलब्ध नहीं है\s*$"),
]

# If ANY of these substrings appear anywhere in a doc's raw text, the
# whole doc is dropped: it's a template/utility page (weather widget,
# subscription FAQ, IPO listing, homepage headline dump), not an
# article. These pages are near-identical across thousands of URLs
# (same boilerplate, only a city name or number changes), so they
# inflate duplicate/templated phrasing far more than real articles do.
DOC_DROP_MARKERS = [
    "मौसम अपडेट — क्या मिलेगा इस पेज पर",   # weather widget pages
    "यह एक डिजिटल सब्सक्रिप्शन है",          # subscription FAQ page
    "IPO वह प्रक्रिया है जिसमें कोई कंपनी",   # IPO listing page
    "होम पेज पर एक नजर में देखें",           # site homepage index dump
    "BBC News, हिंदी - होम पेज",             # BBC homepage index dump
]


# ----------------------------------------------------------------------
# CLEANING
# ----------------------------------------------------------------------
def strip_boilerplate(text: str) -> str:
    for pat in BOILERPLATE_PATTERNS:
        text = pat.sub("", text)
    return text


def line_is_english_or_noise(line: str) -> bool:
    line = line.strip()
    if len(line) < MIN_LINE_CHARS:
        return True
    for pat in NOISE_LINE_PATTERNS:
        if pat.search(line):
            return True
    dev = len(DEVANAGARI_RE.findall(line))
    lat = len(LATIN_LETTER_RE.findall(line))
    total_letters = dev + lat
    if total_letters == 0:
        # no letters at all (pure numbers/symbols/dates) -> drop, it's
        # not useful standalone text for a language-model corpus
        return True
    ratio = dev / total_letters
    return ratio < MIN_LINE_DEVANAGARI_RATIO


def is_junk_page(raw_text: str) -> bool:
    """Whole-document drop: template/utility pages (weather widgets,
    subscription FAQs, IPO listings, homepage headline dumps) are
    fluent Hindi but not articles -- and are near-duplicates of
    thousands of other URLs with only a city name/number changed."""
    return any(marker in raw_text for marker in DOC_DROP_MARKERS)


def clean_doc_text(raw_text: str) -> str:
    text = unicodedata.normalize("NFC", raw_text)
    text = strip_boilerplate(text)

    kept_lines = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        if line_is_english_or_noise(line):
            continue
        kept_lines.append(line)

    # flatten to a single line (no embedded newlines) with single spaces
    flat = " ".join(kept_lines)
    flat = re.sub(r"\s+", " ", flat).strip()
    return flat


def doc_signature(text: str) -> str:
    """Same style as the crawler's near-dup key: first+last 200 chars
    + length band. Catches the same article crawled twice."""
    key = text[:200] + text[-200:] + str(len(text) // 200)
    return hashlib.sha1(key.encode("utf-8")).hexdigest()


def exact_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
def main():
    shard_paths = sorted(glob.glob(os.path.join(INPUT_DIR, "shard_*.jsonl")))
    if not shard_paths:
        print(f"[error] no shard_*.jsonl files found in {INPUT_DIR}")
        return

    print(f"[info] found {len(shard_paths)} shard files")

    seen_exact = set()
    seen_near = set()

    docs_in = 0
    docs_out = 0
    dropped_too_short = 0
    dropped_exact_dup = 0
    dropped_near_dup = 0
    dropped_json_error = 0
    dropped_junk_page = 0

    words_in = 0
    words_out = 0

    out_dir = os.path.dirname(OUTPUT_FILE)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as out_f:
        for shard_path in shard_paths:
            with open(shard_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    docs_in += 1
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        dropped_json_error += 1
                        continue

                    raw_text = rec.get("text", "")
                    words_in += len(WORD_RE.findall(raw_text))

                    if is_junk_page(raw_text):
                        dropped_junk_page += 1
                        continue

                    cleaned = clean_doc_text(raw_text)
                    word_count = len(WORD_RE.findall(cleaned))

                    if word_count < MIN_DOC_WORDS_AFTER_CLEAN:
                        dropped_too_short += 1
                        continue

                    eh = exact_hash(cleaned)
                    if eh in seen_exact:
                        dropped_exact_dup += 1
                        continue

                    nh = doc_signature(cleaned)
                    if nh in seen_near:
                        dropped_near_dup += 1
                        continue

                    seen_exact.add(eh)
                    seen_near.add(nh)

                    out_f.write(cleaned + "\n")
                    docs_out += 1
                    words_out += word_count

            print(f"[progress] processed {os.path.basename(shard_path)} "
                  f"-> docs_out so far: {docs_out:,}")

    print("\n=== DONE ===")
    print(f"Docs in:              {docs_in:,}")
    print(f"Docs out (clean):     {docs_out:,}")
    print(f"  dropped (too short after cleaning): {dropped_too_short:,}")
    print(f"  dropped (exact duplicate):          {dropped_exact_dup:,}")
    print(f"  dropped (near duplicate):           {dropped_near_dup:,}")
    print(f"  dropped (bad JSON line):            {dropped_json_error:,}")
    print(f"  dropped (junk/template page):        {dropped_junk_page:,}")
    print(f"Words in (raw):        {words_in:,}")
    print(f"Words out (clean):     {words_out:,}")
    print(f"Output file: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()