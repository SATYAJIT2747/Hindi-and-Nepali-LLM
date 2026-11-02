"""
Nepali Corpus Preprocessor — same pipeline as your Hindi script, plus
two Nepali-specific additions found in your sample data:
  A) Homepage/index-dump pages (root or category URLs like
     "karobardaily.com/", "bbc.com/nepali") are junk: they're headline
     aggregations, not articles, and duplicate real article text.
     Detected structurally: URL's last path segment has no digit and
     no hyphen -> real articles in your crawl always have one or the
     other ("/news/405822" or "/agriculture-officials-inspect-...").
  B) Within-doc exact-line dedup: some sites (ekantipur) repeat the
     same headline paragraph twice in one scrape (sidebar + main list).
"""

import glob
import hashlib
import json
import os
import re
import unicodedata
from urllib.parse import urlparse

INPUT_DIR = r"C:\Users\satya\Desktop\LMA_PROJ\Nepali\nepali_corpus"
OUTPUT_FILE = r"C:\Users\satya\Desktop\LMA_PROJ\Nepali\nepali_corpus_clean.txt"

MIN_LINE_DEVANAGARI_RATIO = 0.5
MIN_LINE_CHARS = 8
MIN_DOC_WORDS_AFTER_CLEAN = 40

DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
LATIN_LETTER_RE = re.compile(r"[A-Za-z]")
WORD_RE = re.compile(r"\S+")

# Site footer/boilerplate blocks -- add more as you spot them per domain.
BOILERPLATE_PATTERNS = [
    re.compile(r"प्रभात खबर डिजिटल टॉप स्टोरी.*", re.DOTALL),
    re.compile(r"Prabhat Khabar App\s*:.*", re.DOTALL),
]

NOISE_LINE_PATTERNS = [
    re.compile(r"^यह भी पढ़ें"),
    re.compile(r"^Also Read"),
    re.compile(r"^Featured Stories$"),
]

# Whole-doc drop if ANY substring appears (template/index pages).
DOC_DROP_MARKERS = [
    "BBC News,",              # any "BBC News, <lang> - होमपेज" variant
    "Featured Stories",
]


def is_index_page(url: str) -> bool:
    """Homepage/category dump: last URL path segment has neither a
    digit nor a hyphen (real articles in this crawl always have one)."""
    path = urlparse(url).path.strip("/")
    if not path:
        return True
    last = path.split("/")[-1]
    return not (any(c.isdigit() for c in last) or "-" in last)


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
    total = dev + lat
    if total == 0:
        return True
    return (dev / total) < MIN_LINE_DEVANAGARI_RATIO


def is_junk_page(raw_text: str) -> bool:
    return any(marker in raw_text for marker in DOC_DROP_MARKERS)


def clean_doc_text(raw_text: str) -> str:
    text = unicodedata.normalize("NFC", raw_text)
    text = strip_boilerplate(text)

    kept_lines = []
    seen_lines = set()
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        if line_is_english_or_noise(line):
            continue
        if line in seen_lines:      # (B) drop in-doc repeats
            continue
        seen_lines.add(line)
        kept_lines.append(line)

    flat = " ".join(kept_lines)
    flat = re.sub(r"\s+", " ", flat).strip()
    return flat


def doc_signature(text: str) -> str:
    key = text[:200] + text[-200:] + str(len(text) // 200)
    return hashlib.sha1(key.encode("utf-8")).hexdigest()


def exact_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main():
    shard_paths = sorted(glob.glob(os.path.join(INPUT_DIR, "shard_*.jsonl")))
    if not shard_paths:
        print(f"[error] no shard_*.jsonl files found in {INPUT_DIR}")
        return
    print(f"[info] found {len(shard_paths)} shard files")

    seen_exact, seen_near = set(), set()
    docs_in = docs_out = 0
    dropped_too_short = dropped_exact_dup = dropped_near_dup = 0
    dropped_json_error = dropped_junk_page = dropped_index_page = 0
    words_in = words_out = 0

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

                    url = rec.get("url", "")
                    raw_text = rec.get("text", "")
                    words_in += len(WORD_RE.findall(raw_text))

                    if is_index_page(url):
                        dropped_index_page += 1
                        continue
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

            print(f"[progress] {os.path.basename(shard_path)} -> docs_out: {docs_out:,}")

    print("\n=== DONE ===")
    print(f"Docs in:  {docs_in:,}")
    print(f"Docs out: {docs_out:,}")
    print(f"  dropped (index/homepage page):      {dropped_index_page:,}")
    print(f"  dropped (junk/template page):        {dropped_junk_page:,}")
    print(f"  dropped (too short after cleaning):  {dropped_too_short:,}")
    print(f"  dropped (exact duplicate):           {dropped_exact_dup:,}")
    print(f"  dropped (near duplicate):            {dropped_near_dup:,}")
    print(f"  dropped (bad JSON line):             {dropped_json_error:,}")
    print(f"Words in (raw):  {words_in:,}")
    print(f"Words out (clean): {words_out:,}")
    print(f"Output file: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
