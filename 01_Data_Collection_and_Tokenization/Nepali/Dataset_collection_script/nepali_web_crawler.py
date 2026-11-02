"""
Nepali Human-Written Web Crawler — Model L manual-collection component
=========================================================================

Goal: crawl ~100M tokens (~60-70M words) of ORIGINAL, human-written Nepali
text from Nepal-based news / literature / encyclopedic sites — no
machine-translated content — with full checkpoint/resume support.

IMPORTANT — Nepali vs Hindi disambiguation:
Nepali and Hindi share the Devanagari script, so a simple "is this
Devanagari?" check is NOT enough — it would happily accept Hindi text too.
This script adds a second filter: a frequency check on Nepali-specific
function words/verb forms (छ, छन्, भएको, गरेको, थियो...) that are
characteristic of Nepali grammar and rare in Hindi. Combined with
restricting crawl domains to Nepal-based sites, this keeps the corpus clean.

SETUP:
    uv add trafilatura aiohttp aiodns beautifulsoup4 nest_asyncio
    (or: pip install trafilatura aiohttp aiodns beautifulsoup4 nest_asyncio)

RUN:
    uv run python nepali_web_crawler.py

RESUME: just re-run the same command — reads checkpoint.json in OUTPUT_DIR
and continues where it left off.
"""

import asyncio
import aiohttp
import hashlib
import json
import os
import random
import re
import time
import urllib.robotparser as robotparser
from collections import deque
from urllib.parse import urljoin, urlparse

import trafilatura
from bs4 import BeautifulSoup
from aiohttp.resolver import ThreadedResolver

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
# NOTE: hardcoded to your local machine, NOT Kaggle. This points at a
# NEW folder so this fresh 40M-token top-up run doesn't touch/overwrite
# your archived nepali_corpus data. Merge the two shard sets afterward.
OUTPUT_DIR = r"C:\Users\satya\Desktop\LMA_PROJ\Nepali\nepali_corpus_v2"

DEBUG = os.environ.get("DEBUG", "0") == "1"  # set DEBUG=1 to see fetch errors
SHARD_MAX_DOCS = 2000          # docs per output shard file
TARGET_TOKENS = 40_000_000     # this run: top up ~40M fresh tokens
                                # (your archived earlier run already has
                                # ~59M -> combine both later for ~100M)
MAX_WALLCLOCK_HOURS = 7.5      # safety stop for long unattended runs
CONCURRENCY = 48               # concurrent HTTP requests (global)
PER_DOMAIN_DELAY = 1.0         # min seconds between requests to same domain
REQUEST_TIMEOUT = 15
MIN_DEVANAGARI_RATIO = 0.55    # filter out pages that aren't Devanagari at all
MIN_NEPALI_MARKER_RATIO = 0.006   # >= ~1 Nepali marker word per 165 words
MIN_DOC_WORDS = 80             # discard tiny/boilerplate pages
MAX_RETRIES = 3                # attempts before giving up on a URL
CHECKPOINT_EVERY_N_DOCS = 100  # how often to checkpoint to disk

USER_AGENT = (
    "Mozilla/5.0 (compatible; ResearchCorpusBot/1.0; "
    "+https://example.edu/research-project; academic dataset collection)"
)

# Seed list: Nepal-based, human-authored sources only.
SEED_URLS = [
    # News (professional editorial newsrooms based in Nepal) — original set
    "https://ekantipur.com/",
    "https://www.onlinekhabar.com/",
    "https://www.setopati.com/",
    "https://nagariknews.nagariknetwork.com/",
    "https://www.ratopati.com/",
    "https://www.bbc.com/nepali",
    "https://www.dw.com/ne/",
    "https://www.nayapatrikadaily.com/",
    "https://annapurnapost.com/",
    # News — added for wider domain coverage / throughput
    # (checked live: dropped khabarhub.com, gorkhapatraonline.com,
    # lokaantar.com, sahityapost.com — all failed to respond)
    "https://www.nepalpress.com/",
    "https://www.himalkhabar.com/",
    "https://www.nepalkhabar.com/",
    "https://baahrakhari.com/",
    "https://thahakhabar.com/",
    "https://www.karobardaily.com/",
    "https://www.rajdhanidaily.com/",
    "https://www.dcnepal.com/",
    "https://ujyaaloonline.com/",
    "https://www.nepalnews.com/",
    # Literature / long-form human writing
    "https://www.nepalisahitya.com/",
    # Encyclopedic (human-edited, not MT)
    "https://ne.wikipedia.org/wiki/मुख्य_पृष्ठ",
    "https://ne.wikisource.org/wiki/मुख्य_पृष्ठ",
]

ALLOWED_DOMAINS = {urlparse(u).netloc for u in SEED_URLS}

DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
WORD_RE = re.compile(r"\S+")

# Function words / verb forms distinctive of Nepali (rare/absent in Hindi).
# "छ/छन्/छु/छौं" (copula "is/are"), "भएको/गरेको/थियो" (participles/past).
# "पनि/लागि/बाट" are shared with Hindi so deliberately excluded, to keep
# the marker set precise rather than broad.
NEPALI_MARKERS = [
    "छ", "छन्", "छु", "छौं", "छैन", "भएको", "गरेको", "गर्दै",
    "थियो", "थिए", "हुन्छ", "गर्नुहुन्छ", "रहेको", "भन्नुभयो",
]
NEPALI_MARKER_RE = re.compile(
    r"(?:^|\s)(" + "|".join(map(re.escape, NEPALI_MARKERS)) + r")(?:\s|$)"
)


# ----------------------------------------------------------------------
# UTILITIES
# ----------------------------------------------------------------------
def normalize_text(text: str) -> str:
    import unicodedata
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def devanagari_ratio(text: str) -> float:
    if not text:
        return 0.0
    dev = len(DEVANAGARI_RE.findall(text))
    total = len(re.findall(r"\S", text))
    return dev / max(total, 1)


def nepali_marker_ratio(text: str, word_count: int) -> float:
    if word_count == 0:
        return 0.0
    hits = len(NEPALI_MARKER_RE.findall(text))
    return hits / word_count


def estimate_tokens(text: str) -> int:
    """Rough proxy; recompute exactly once your Phase-1 tokenizer exists."""
    words = len(WORD_RE.findall(text))
    return int(words * 1.7)


def doc_hash(text: str) -> str:
    key = text[:200] + text[-200:] + str(len(text) // 200)
    return hashlib.sha1(key.encode("utf-8")).hexdigest()


# ----------------------------------------------------------------------
# CRAWLER STATE (checkpointable)
# ----------------------------------------------------------------------
class CrawlState:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.ckpt_path = os.path.join(output_dir, "checkpoint.json")
        self.visited = set()
        self.seen_hashes = set()
        self.frontier = deque()
        self.fail_counts = {}
        self.domain_last_hit = {}
        self.robots_cache = {}
        self.total_tokens = 0
        self.total_docs = 0
        self.shard_idx = 0
        self.shard_buf = []
        self.start_time = time.time()
        self.load()

    def load(self):
        if os.path.exists(self.ckpt_path):
            with open(self.ckpt_path, "r", encoding="utf-8") as f:
                ck = json.load(f)
            self.visited = set(ck["visited"])
            self.seen_hashes = set(ck["seen_hashes"])
            self.frontier = deque(ck["frontier"])
            self.total_tokens = ck["total_tokens"]
            self.total_docs = ck["total_docs"]
            self.shard_idx = ck["shard_idx"]
            print(f"[resume] loaded checkpoint: {self.total_docs} docs, "
                  f"{self.total_tokens:,} tokens so far")
        else:
            self.frontier.extend(SEED_URLS)
            print("[fresh start] no checkpoint found, seeding frontier")

    # ---- FIX #1: checkpointing no longer forces a shard flush every
    # 100 docs. In the old code, save() was called every 100 docs AND
    # always flushed shard_buf, so shard_buf never reached SHARD_MAX_DOCS
    # (2000) and shard_idx never incremented -> everything piled into
    # shard_000000.jsonl forever. Now: frequent lightweight checkpoint
    # writes, full flush only on real shard rollover or clean shutdown.
    def _save_checkpoint_only(self):
        ck = {
            "visited": list(self.visited)[-2_000_000:],
            "seen_hashes": list(self.seen_hashes)[-2_000_000:],
            "frontier": list(self.frontier)[:50_000],
            "total_tokens": self.total_tokens,
            "total_docs": self.total_docs,
            "shard_idx": self.shard_idx,
        }
        tmp = self.ckpt_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(ck, f)
        os.replace(tmp, self.ckpt_path)  # atomic write

    def save(self):
        """Full save: flush shard buffer AND write checkpoint. Used at
        clean shutdown (end of main loop) so nothing is left unflushed."""
        self._flush_shard()
        self._save_checkpoint_only()

    def _flush_shard(self):
        if not self.shard_buf:
            return
        path = os.path.join(
            self.output_dir, f"shard_{self.shard_idx:06d}.jsonl"
        )
        with open(path, "a", encoding="utf-8") as f:
            for rec in self.shard_buf:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self.shard_buf = []

    def add_doc(self, rec: dict):
        self.shard_buf.append(rec)
        self.total_docs += 1
        self.total_tokens += rec["est_tokens"]
        if len(self.shard_buf) >= SHARD_MAX_DOCS:
            self._flush_shard()
            self.shard_idx += 1
        if self.total_docs % CHECKPOINT_EVERY_N_DOCS == 0:
            self._save_checkpoint_only()
            elapsed = (time.time() - self.start_time) / 3600
            print(f"[{elapsed:.2f}h] docs={self.total_docs:,} "
                  f"tokens~{self.total_tokens:,} "
                  f"frontier={len(self.frontier):,}")


# ----------------------------------------------------------------------
# ROBOTS.TXT HANDLING
# ----------------------------------------------------------------------
async def get_robots(session, state: CrawlState, domain: str):
    if domain in state.robots_cache:
        return state.robots_cache[domain]
    rp = robotparser.RobotFileParser()
    url = f"https://{domain}/robots.txt"
    try:
        async with session.get(url, timeout=REQUEST_TIMEOUT) as resp:
            text = await resp.text(errors="ignore")
        rp.parse(text.splitlines())
    except Exception:
        rp = None
    state.robots_cache[domain] = rp
    return rp


def robots_allows(rp, url: str) -> bool:
    if rp is None:
        return True
    try:
        return rp.can_fetch(USER_AGENT, url)
    except Exception:
        return True


# ----------------------------------------------------------------------
# FETCH + EXTRACT + LINK DISCOVERY
# ----------------------------------------------------------------------
async def polite_wait(state: CrawlState, domain: str):
    last = state.domain_last_hit.get(domain, 0)
    wait = PER_DOMAIN_DELAY - (time.time() - last)
    if wait > 0:
        await asyncio.sleep(wait)
    state.domain_last_hit[domain] = time.time()


async def fetch(session, url: str):
    headers = {"User-Agent": USER_AGENT}
    try:
        async with session.get(
            url, headers=headers, timeout=REQUEST_TIMEOUT
        ) as resp:
            if resp.status != 200:
                return None
            ctype = resp.headers.get("Content-Type", "")
            if "text/html" not in ctype:
                return None
            return await resp.text(errors="ignore")
    except Exception as e:
        if DEBUG:
            print(f"[fetch-fail] {url} -> {type(e).__name__}: {e}")
        return None


def extract_links(html: str, base_url: str, domain: str):
    links = []
    try:
        soup = BeautifulSoup(html, "html.parser")
        for a in soup.find_all("a", href=True):
            href = urljoin(base_url, a["href"])
            href = href.split("#")[0]
            p = urlparse(href)
            if p.scheme not in ("http", "https"):
                continue
            if p.netloc != domain:
                continue
            if re.search(
                r"\.(jpg|jpeg|png|gif|pdf|zip|mp4|mp3|css|js)$",
                p.path,
                re.I,
            ):
                continue
            links.append(href)
    except Exception:
        pass
    return links


async def process_url(session, state: CrawlState, url: str, sem):
    domain = urlparse(url).netloc
    if domain not in ALLOWED_DOMAINS:
        return
    if url in state.visited:
        return

    async with sem:
        rp = await get_robots(session, state, domain)
        if not robots_allows(rp, url):
            state.visited.add(url)  # permanently skip (policy), not a retry case
            return
        await polite_wait(state, domain)
        html = await fetch(session, url)

    if not html:
        # Transient failure -> requeue for retry instead of marking
        # visited forever, so a temporary network blip doesn't drain
        # the whole frontier to empty.
        state.fail_counts[url] = state.fail_counts.get(url, 0) + 1
        if state.fail_counts[url] < MAX_RETRIES:
            state.frontier.append(url)
        else:
            state.visited.add(url)  # give up after MAX_RETRIES
        return

    state.visited.add(url)  # only now, after a successful fetch

    for link in extract_links(html, url, domain):
        if link not in state.visited:
            state.frontier.append(link)

    text = trafilatura.extract(
        html, include_comments=False, include_tables=False,
        favor_precision=True,
    )
    if not text:
        return
    text = normalize_text(text)

    word_count = len(WORD_RE.findall(text))
    if word_count < MIN_DOC_WORDS:
        return
    if devanagari_ratio(text) < MIN_DEVANAGARI_RATIO:
        return
    if nepali_marker_ratio(text, word_count) < MIN_NEPALI_MARKER_RATIO:
        # Devanagari but doesn't look grammatically Nepali -> likely a
        # Hindi widget/embed/syndicated snippet on the page. Skip it.
        return

    h = doc_hash(text)
    if h in state.seen_hashes:
        return
    state.seen_hashes.add(h)

    rec = {
        "url": url,
        "domain": domain,
        "text": text,
        "word_count": word_count,
        "est_tokens": estimate_tokens(text),
        "crawl_ts": time.time(),
    }
    state.add_doc(rec)


# ----------------------------------------------------------------------
# MAIN LOOP
# ----------------------------------------------------------------------
async def main():
    state = CrawlState(OUTPUT_DIR)
    sem = asyncio.Semaphore(CONCURRENCY)

    # ThreadedResolver avoids a Windows-specific bug where aiodns'
    # AsyncResolver fails every DNS lookup instantly under the default
    # event loop, silently draining the whole frontier in <1 second.
    connector = aiohttp.TCPConnector(
        limit=CONCURRENCY, ssl=False, resolver=ThreadedResolver()
    )
    async with aiohttp.ClientSession(connector=connector) as session:
        try:
            while state.frontier:
                elapsed_h = (time.time() - state.start_time) / 3600
                if elapsed_h >= MAX_WALLCLOCK_HOURS:
                    print("[stop] wallclock limit reached, checkpointing...")
                    break
                if state.total_tokens >= TARGET_TOKENS:
                    print("[stop] token target reached, checkpointing...")
                    break

                batch = []
                while state.frontier and len(batch) < CONCURRENCY * 2:
                    batch.append(state.frontier.popleft())

                random.shuffle(batch)

                tasks = [process_url(session, state, u, sem) for u in batch]
                await asyncio.gather(*tasks, return_exceptions=True)
        finally:
            # ---- FIX #2: save on ANY exit path, including Ctrl+C /
            # KeyboardInterrupt, not just a clean while-loop finish.
            # Previously, a KeyboardInterrupt during the gather() call
            # skipped straight past `state.save()` at the bottom of
            # main(), so up to CHECKPOINT_EVERY_N_DOCS-1 docs sitting in
            # shard_buf were silently discarded on every interrupt.
            print("[shutdown] saving checkpoint + flushing shard buffer...")
            state.save()

    elapsed = (time.time() - state.start_time) / 3600
    print(f"\n=== DONE ===\nTime: {elapsed:.2f}h | Docs: {state.total_docs:,} "
          f"| Est. tokens: {state.total_tokens:,}")
    print(f"Output shards in: {OUTPUT_DIR}")


if __name__ == "__main__":
    try:
        asyncio.get_running_loop()
        # Already inside a running loop -> Jupyter/Kaggle-style env,
        # where plain asyncio.run() would raise "cannot be called from
        # a running event loop". Patch it so it can nest.
        import nest_asyncio
        nest_asyncio.apply()
    except RuntimeError:
        # No running loop -> plain script execution (e.g. `uv run
        # python file.py`). nest_asyncio must NOT be applied here:
        # patching an already-normal loop corrupts aiohttp's internal
        # timeout/task tracking and causes every request to fail with
        # "Timeout context manager should be used inside a task".
        pass

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[interrupted] Ctrl+C received — checkpoint was already "
              "saved inside main()'s finally block. Safe to re-run the "
              "same command to resume.")
