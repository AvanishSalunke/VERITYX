"""Shared paths and loaders for the eval harness."""
import gzip
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "Train" / "Train"
DOCS_DIR = DATA / "Documents"
BENCHMARK = DATA / "Benchmarks" / "benchmark_100.jsonl"

EVAL = Path(__file__).resolve().parent
CACHE = EVAL / "cache"
PAGES_FILE = CACHE / "pages.jsonl.gz"   # one row per PDF page: {doc, page (1-based), text}
GOLD_FILE = CACHE / "gold.json"         # per-question gold docs/pages/numbers
ALIASES_FILE = CACHE / "doc_aliases.json"  # gold doc name -> on-disk PDF name
SPLITS_FILE = EVAL / "splits.json"      # fixed dev / holdout question ids
RUNS = EVAL / "runs"
RESULTS_CSV = EVAL / "results.csv"


def load_benchmark():
    with open(BENCHMARK, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_pages():
    with gzip.open(PAGES_FILE, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def load_gold():
    return json.loads(GOLD_FILE.read_text(encoding="utf-8"))


def load_splits():
    return json.loads(SPLITS_FILE.read_text(encoding="utf-8"))


_MD_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_MD_MARKS = re.compile(r"[*#|_`>]+")
_SPACE = re.compile(r"\s+")


def normalize(text):
    """Strip markdown markup and collapse whitespace so gold chunks compare with raw PDF text."""
    text = _MD_IMAGE.sub(" ", text or "")
    text = _MD_MARKS.sub(" ", text)
    return _SPACE.sub(" ", text).strip().lower()


_CITATION = re.compile(r"\[[^\]]*\]")
_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")


def extract_numbers(text):
    """Distinct numeric tokens (commas stripped, >=2 digits), ignoring [citation] brackets."""
    text = _CITATION.sub(" ", text or "")
    out = set()
    for tok in _NUMBER.findall(text):
        tok = tok.replace(",", "").rstrip(".")
        if sum(ch.isdigit() for ch in tok) >= 2:
            out.add(tok.lstrip("0") or "0")
    return out
