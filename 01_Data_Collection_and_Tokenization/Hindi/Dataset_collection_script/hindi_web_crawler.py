"""
Hindi Human-Written Web Crawler — Model H manual-collection component
=======================================================================

Goal: crawl ~100M tokens (~65-75M words) of ORIGINAL, human-written Hindi
text from reputable news / literature / blog sites — no machine-translated
or aggregator content — with full checkpoint/resume support.

WHY THESE SITES: all are known editorial/human-authored Hindi content
(professional newsrooms or literary archives), not MT pipelines.

SETUP:
    uv add trafilatura aiohttp aiodns beautifulsoup4 nest_asyncio
    (or: pip install trafilatura aiohttp aiodns beautifulsoup4 nest_asyncio)

RUN:
    uv run python hindi_web_crawler.py

RESUME: just re-run the same command. It reads checkpoint.json in
OUTPUT_DIR and continues where it left off.
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
# NOTE: hardcoded to your local machine, NOT Kaggle. Update this path if
# your project folder ever moves. Mirrors the Nepali project layout.
OUTPUT_DIR = r"C:\Users\satya\Desktop\LMA_PROJ\Nepali\hindi_corpus"

DEBUG = os.environ.get("DEBUG", "0") == "1"  # set DEBUG=1 to see fetch errors
SHARD_MAX_DOCS = 2000          # docs per output shard file
TARGET_TOKENS = 100_000_000    # stop condition (rough estimate, see below)
MAX_WALLCLOCK_HOURS = 7.5      # safety stop for long unattended runs
CONCURRENCY = 48               # concurrent HTTP requests (global)
PER_DOMAIN_DELAY = 1.0         # min seconds between requests to same domain
REQUEST_TIMEOUT = 15
MIN_DEVANAGARI_RATIO = 0.55    # filter out pages that aren't really Hindi
MIN_DOC_WORDS = 80             # discard tiny/boilerplate pages
MAX_RETRIES = 3                # attempts before giving up on a URL
CHECKPOINT_EVERY_N_DOCS = 100  # how often to checkpoint to disk

USER_AGENT = (
    "Mozilla/5.0 (compatible; ResearchCorpusBot/1.0; "
    "+https://example.edu/research-project; academic dataset collection)"
)

# Seed list: reputable human-authored Hindi sources.
SEED_URLS = [
    # News (editorial, human-written) — original set
    "https://www.bbc.com/hindi",
    "https://www.dw.com/hi/",
    "https://hindi.newslaundry.com/",
    "https://satyagrah.scroll.in/",
    "https://www.prabhatkhabar.com/",
    "https://www.jagran.com/",
    "https://navbharattimes.indiatimes.com/",
    "https://www.livehindustan.com/",
    "https://www.amarujala.com/",
    "https://www.patrika.com/",
    # News — added for wider domain coverage / throughput
    "https://www.aajtak.in/",
    "https://ndtv.in/",
    "https://zeenews.india.com/hindi",
    "https://hindi.news18.com/",
    "https://www.abplive.com/",
    "https://www.indiatv.in/",
    "https://www.thehindu.com/hindi/",
    "https://www.deshbandhu.co.in/",
    "https://www.rajasthanpatrika.com/",
    "https://www.punjabkesari.in/",
    "https://www.hindustantimes.com/hindi",
    "https://hindi.oneindia.com/",
    "https://hindi.asianetnews.com/",
    # Magazines / long-form editorial (essays, opinion, culture)
    "https://hindi.outlookindia.com/",
    "https://www.bhaskar.com/",
    "https://www.dainikbhaskar.com/",
    "https://www.pratilipi.com/",
    # Literature / long-form (human-authored, high lexical diversity)
    "https://www.kavitakosh.org/kk/",
    "https://www.gadyakosh.org/gk/",
    "https://hindi.sahityakunj.net/",
    "https://www.hindisamay.com/",
    "https://www.rekhta.org/",
    # Encyclopedic (human-edited, not MT)
    "https://hi.wikipedia.org/wiki/मुखपृष्ठ",
    "https://hi.wikisource.org/wiki/मुखपृष्ठ",
]

ALLOWED_DOMAINS = {urlparse(u).netloc for u in SEED_URLS}

DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
WORD_RE = re.compile(r"\S+")


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

    # ---- FIX #1: checkpointing no longer force-flushes the shard every
    # 100 docs, so shard_buf can actually reach SHARD_MAX_DOCS (2000) and
    # roll over into shard_000001.jsonl, shard_000002.jsonl, etc.
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

    if len(WORD_RE.findall(text)) < MIN_DOC_WORDS:
        return
    if devanagari_ratio(text) < MIN_DEVANAGARI_RATIO:
        return

    h = doc_hash(text)
    if h in state.seen_hashes:
        return
    state.seen_hashes.add(h)

    rec = {
        "url": url,
        "domain": domain,
        "text": text,
        "word_count": len(WORD_RE.findall(text)),
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
            # KeyboardInterrupt, not just a clean while-loop finish. This
            # guarantees a checkpoint exists no matter how the run ends.
            print("[shutdown] saving checkpoint + flushing shard buffer...")
            state.save()

    elapsed = (time.time() - state.start_time) / 3600
    print(f"\n=== DONE ===\nTime: {elapsed:.2f}h | Docs: {state.total_docs:,} "
          f"| Est. tokens: {state.total_tokens:,}")
    print(f"Output shards in: {OUTPUT_DIR}")


if __name__ == "__main__":
    try:
        asyncio.get_running_loop()
        import nest_asyncio
        nest_asyncio.apply()
    except RuntimeError:
        # No running loop -> plain script execution. nest_asyncio must
        # NOT be applied here: patching an already-normal loop corrupts
        # aiohttp's internal timeout/task tracking.
        pass

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[interrupted] Ctrl+C received — checkpoint was already "
              "saved inside main()'s finally block. Safe to re-run the "
              "same command to resume.")
