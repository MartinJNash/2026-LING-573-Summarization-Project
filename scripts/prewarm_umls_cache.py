"""
Pre-warm the UTS API cache for all test-set documents before inference.

Runs scispaCy NER on every document to extract candidate medical terms,
then fires UMLS/RxNorm lookups for each unique term and stores the result in
uts_cache.sqlite.  The cache is shared with the inference pipeline, so when
the array job runs later every lookup is a local SQLite read (~0 ms) instead
of a live API call (~1-2 s).

Usage (on Hyak, after git pull):
    source /gscratch/scrubbed/<netid>/medjargone/bin/activate
    export UMLS_API_KEY=<your_key>
    PYTHONPATH=src python scripts/prewarm_umls_cache.py
    PYTHONPATH=src python scripts/prewarm_umls_cache.py --split train
"""

import argparse
import sys
import time
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from medjargone import config
from medjargone.pipeline.glossary import (
    _extract_candidates, _gloss_one,
    APICache, UMLSIndex, UTSClient,
)


def _read_folder(root: Path):
    for ft_path in sorted((root / "fulltext").iterdir()):
        if ft_path.is_file():
            yield ft_path.read_text(encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Pre-warm UMLS API cache")
    parser.add_argument("--split", choices=["test", "train"], default="test")
    parser.add_argument("--max-docs", type=int, default=None,
                        help="Cap number of documents (for testing)")
    args = parser.parse_args()

    if not config.UMLS_API_KEY:
        print("[error] UMLS_API_KEY not set — export UMLS_API_KEY=<key>")
        sys.exit(1)

    data_dir = (config.MULTICLINSUM_TEST_DIR
                if args.split == "test"
                else config.MULTICLINSUM_TRAIN_DIR)
    if not data_dir.exists():
        print(f"[error] Dataset not found at {data_dir}")
        sys.exit(1)

    cache     = APICache()
    uts       = UTSClient(cache=cache)
    umls_idx  = UMLSIndex() if config.UMLS_SUBSET_DB.exists() else None

    if umls_idx:
        print("[info] Local UMLS subset found — cache warm-up mostly redundant "
              "(local DB is already fast). Continuing anyway to populate RxNorm cache.")
    else:
        print("[info] No local UMLS subset — will use UTS API (needs UMLS_API_KEY).")

    docs = list(_read_folder(data_dir))
    if args.max_docs:
        docs = docs[:args.max_docs]
    print(f"Loaded {len(docs)} documents from {data_dir}")

    # ── Pass 1: NER all documents, collect unique (term, ner_type) pairs ──────
    print("\nPass 1: extracting candidates via scispaCy NER…")
    unique_terms: dict[str, str] = {}   # term_lc → ner_type (first seen wins)
    abbrev_maps: list[dict] = []

    for i, text in enumerate(docs):
        if (i + 1) % 100 == 0:
            print(f"  NER {i+1}/{len(docs)} docs  |  {len(unique_terms)} unique terms so far")
        candidates, abbrev_map = _extract_candidates(text)
        abbrev_maps.append(abbrev_map)
        for term, ner_type, *_ in candidates:
            key = term.lower()
            if key not in unique_terms:
                unique_terms[key] = ner_type

    print(f"\nPass 1 done: {len(unique_terms)} unique candidate terms across {len(docs)} docs")

    # ── Pass 2: UMLS/RxNorm lookup for each unique term ───────────────────────
    print("\nPass 2: looking up unique terms in UMLS (cache misses → UTS API)…")
    n_cached = 0
    n_fetched = 0
    n_failed  = 0
    curated: dict[str, str] = {}
    if config.CURATED_ANATOMY.exists():
        import json
        curated = json.load(open(config.CURATED_ANATOMY))

    terms_list = list(unique_terms.items())
    t_start = time.monotonic()

    for i, (term, ner_type) in enumerate(terms_list):
        if (i + 1) % 200 == 0:
            elapsed = time.monotonic() - t_start
            rate = (i + 1) / elapsed
            eta  = (len(terms_list) - i - 1) / rate if rate > 0 else 0
            print(f"  {i+1}/{len(terms_list)}  |  "
                  f"fetched={n_fetched} cached={n_cached} failed={n_failed}  |  "
                  f"ETA {eta/60:.1f} min")

        # Check if already cached (from a previous run or from running prelim jobs)
        import urllib.parse
        search_key = config.UTS_SEARCH_URL + "?" + urllib.parse.urlencode(
            sorted({"string": term, "searchType": "exact",
                    "returnIdType": "concept", "pageSize": 5,
                    "sabs": UTSClient._MEDICAL_SABS,
                    "apiKey": config.UMLS_API_KEY}.items())
        )
        if cache.get(search_key) is not None:
            n_cached += 1
            continue

        try:
            _gloss_one(term, ner_type, term, umls_idx, uts, cache, {}, curated)
            n_fetched += 1
        except Exception as exc:
            n_failed += 1
            if n_failed <= 5:
                print(f"  [warn] {term!r}: {exc}")

    elapsed = time.monotonic() - t_start
    print(f"\nPass 2 done in {elapsed/60:.1f} min")
    print(f"  Already cached : {n_cached}")
    print(f"  Newly fetched  : {n_fetched}")
    print(f"  Failed         : {n_failed}")
    print(f"\nCache file: {config.UTS_CACHE_DB}")


if __name__ == "__main__":
    main()
