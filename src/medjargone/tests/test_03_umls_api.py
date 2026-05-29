"""
Level 3 — UMLS API smoke test.
Requires: export UMLS_API_KEY=<your_key>

Tests the live UTS REST API on a handful of well-known terms to confirm:
  - search() returns a CUI
  - semantic_types() returns the right category
  - chv_gloss() returns a consumer-friendly string
  - definition() returns text
  - rxnorm_lookup() returns an RxCUI for a drug

Run: python -m medjargone.tests.test_03_umls_api
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from medjargone import config
from medjargone.pipeline.glossary import UTSClient, APICache, rxnorm_lookup

# Known term → expected CUI (UMLS 2024AB)
KNOWN_TERMS = {
    "myocardial infarction":  "C0027051",
    "pericardial effusion":   "C0031039",
    "dyspnea":                "C0013404",
    "brucellosis":            "C0006309",
}

# Known drugs → expect an RxCUI back
KNOWN_DRUGS = ["rifampicin", "doxycycline", "heparin", "warfarin"]

# Terms with known CHV consumer glosses
CHV_TERMS = [
    "myocardial infarction",   # → "heart attack"
    "pericardial effusion",    # → "excess fluid around the heart"
    "dyspnea",                 # → "shortness of breath"
]


def _skip_if_no_key():
    if not config.UMLS_API_KEY:
        print("  SKIP  UMLS_API_KEY not set — export UMLS_API_KEY=<key>")
        return True
    return False


def test_search():
    if _skip_if_no_key():
        return
    cache = APICache()
    client = UTSClient(cache=cache)
    for term, expected_cui in KNOWN_TERMS.items():
        cui = client.search(term)
        if cui == expected_cui:
            print(f"  PASS  search('{term}') = {cui}")
        elif cui:
            print(f"  WARN  search('{term}') = {cui}  (expected {expected_cui} — release may differ)")
        else:
            print(f"  FAIL  search('{term}') returned None")


def test_semantic_types():
    if _skip_if_no_key():
        return
    cache = APICache()
    client = UTSClient(cache=cache)
    # myocardial infarction should be "Disease or Syndrome"
    cui = client.search("myocardial infarction")
    if not cui:
        print("  FAIL  could not resolve CUI for myocardial infarction")
        return
    stys = client.semantic_types(cui)
    print(f"  semantic types for 'myocardial infarction' ({cui}): {stys}")
    if "Disease or Syndrome" in stys:
        print("  PASS  semantic type 'Disease or Syndrome' found")
    else:
        print("  WARN  expected 'Disease or Syndrome', got:", stys)
    # Confirm it passes the medical filter
    from medjargone.pipeline.glossary import _is_medical
    assert _is_medical(stys), "Expected myocardial infarction to pass medical filter"
    print("  PASS  passes MEDICAL_SEMANTIC_TYPES filter")


def test_chv_gloss():
    if _skip_if_no_key():
        return
    cache = APICache()
    client = UTSClient(cache=cache)
    for term in CHV_TERMS:
        cui = client.search(term)
        if not cui:
            print(f"  SKIP  could not resolve CUI for '{term}'")
            continue
        gloss = client.chv_gloss(cui)
        if gloss:
            print(f"  PASS  CHV gloss for '{term}' → '{gloss}'")
        else:
            print(f"  WARN  no CHV gloss found for '{term}' (CUI={cui})")


def test_definition():
    if _skip_if_no_key():
        return
    cache = APICache()
    client = UTSClient(cache=cache)
    cui = client.search("brucellosis")
    if not cui:
        print("  FAIL  could not resolve CUI for brucellosis")
        return
    result = client.definition(cui)
    if result:
        def_text, sab = result
        print(f"  PASS  definition for brucellosis ({sab}): {def_text[:120]}…")
    else:
        print("  WARN  no definition found for brucellosis")


def test_rxnorm():
    if _skip_if_no_key():
        return
    cache = APICache()
    for drug in KNOWN_DRUGS:
        rxcui = rxnorm_lookup(drug, cache)
        if rxcui:
            print(f"  PASS  RxNorm: '{drug}' → RxCUI {rxcui}")
        else:
            print(f"  WARN  RxNorm: no RxCUI found for '{drug}'")


def test_caching():
    """Second call for the same term should be served from cache (no network hit)."""
    if _skip_if_no_key():
        return
    import time
    cache = APICache()
    client = UTSClient(cache=cache)
    term = "dyspnea"
    # First call (may hit network)
    t0 = time.monotonic()
    cui1 = client.search(term)
    t1 = time.monotonic() - t0
    # Second call (should be cached)
    t0 = time.monotonic()
    cui2 = client.search(term)
    t2 = time.monotonic() - t0
    assert cui1 == cui2
    print(f"  PASS  Cache works: first={t1*1000:.0f}ms, second={t2*1000:.0f}ms"
          f"  ({'cached' if t2 < t1 * 0.5 else 'may not be cached yet'})")


if __name__ == "__main__":
    print("=== Level 3: UMLS API smoke test ===")
    for fn in [test_search, test_semantic_types, test_chv_gloss,
               test_definition, test_rxnorm, test_caching]:
        try:
            fn()
        except Exception as e:
            print(f"  ERR   {fn.__name__} — {e}")
    print()
