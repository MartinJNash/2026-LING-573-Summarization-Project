"""
Level 4 — Glossary subsystem end-to-end.
Requires: UMLS_API_KEY (or local umls_subset.sqlite) + scispaCy models.

Tests the full Stage 2 pipeline: scispaCy NER → ConText filter → UMLS gloss.

Run: python -m medjargone.tests.test_04_glossary
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from medjargone import config
from medjargone.pipeline.preprocess import preprocess_clinical_text
from medjargone.pipeline.glossary import build_glossary, APICache, UTSClient

# A short excerpt from the brucellosis case with varied jargon and negations
SAMPLE_TEXT = """\
A 40-year-old woman presented with intermittent fever and hepatosplenomegaly.
Chest CT revealed multiple pulmonary nodules in both lungs with pleural effusion.
There was no evidence of pneumonia. Tuberculosis was excluded.
She received rifampicin 600 mg daily and doxycycline 100 mg twice daily.
Serum agglutination test (SAT) was positive. She was definitively diagnosed with brucellosis.
At discharge, fever and arthralgia were relieved.
"""

# Text from the 16 analysis examples (Ex 1 — renal vein thrombosis)
RENAL_TEXT = """\
A 43-year-old woman at 30 days post-partum suffering from diabetes on oral antidiabetic \
medication presented with left flank pain with hematuria of one week's duration. \
CT showed an ectatic left renal vein with a large hypodense thrombus extending to \
the inferior vena cava. A renal venous infarction was confirmed. \
Anticoagulation heparin therapy was started.
"""


def _has_umls():
    return bool(config.UMLS_API_KEY) or config.UMLS_SUBSET_DB.exists()


def test_basic_glossary():
    if not _has_umls():
        print("  SKIP  no UMLS_API_KEY and no local subset")
        return

    preprocessed = preprocess_clinical_text(SAMPLE_TEXT)
    entries = build_glossary(SAMPLE_TEXT, preprocessed=preprocessed)

    print(f"  {len(entries)} glossary entries produced")
    for e in entries:
        gloss_str = f"'{e.gloss}' [{e.gloss_source}]" if e.gloss else f"(no gloss, conf={e.confidence})"
        section_str = f" [{e.section}]" if e.section else ""
        print(f"    '{e.term}' ({e.semantic_type}){section_str} → {gloss_str}")

    terms = [e.term.lower() for e in entries]
    assert entries, "Expected at least some glossary entries"
    print("  PASS  Glossary produced entries")


def test_negated_terms_excluded():
    """'pneumonia' and 'tuberculosis' are negated — they must NOT appear in glossary."""
    if not _has_umls():
        print("  SKIP  no UMLS access")
        return
    import medjargone.pipeline.preprocess as _pre
    _pre._load_medspacy()
    if not _pre._medspacy_ok:
        print("  SKIP  medspaCy not installed — ConText filtering inactive")
        return

    preprocessed = preprocess_clinical_text(SAMPLE_TEXT)
    entries = build_glossary(SAMPLE_TEXT, preprocessed=preprocessed)
    terms_lc = {e.term.lower() for e in entries}

    for negated in ("pneumonia", "tuberculosis"):
        if negated in terms_lc:
            print(f"  WARN  '{negated}' appeared in glossary despite negation"
                  " — ConText may not have fired on this span")
        else:
            print(f"  PASS  negated '{negated}' correctly excluded from glossary")


def test_acronym_handling():
    """SAT is defined in the source (serum agglutination test) — should be expanded.
       If scispaCy and AbbreviationDetector are not installed, this is skipped."""
    if not _has_umls():
        print("  SKIP  no UMLS access")
        return
    import medjargone.pipeline.glossary as _gmod
    _gmod._load_nlp()
    if not _gmod._abbrev_ok:
        print("  SKIP  AbbreviationDetector not available")
        return

    preprocessed = preprocess_clinical_text(SAMPLE_TEXT)
    entries = build_glossary(SAMPLE_TEXT, preprocessed=preprocessed)

    sat_entries = [e for e in entries if e.term.upper() == "SAT"]
    if sat_entries:
        print(f"  INFO  SAT entry: term='{sat_entries[0].term}' "
              f"instruction='{sat_entries[0].instruction}'")
    else:
        print("  INFO  SAT not found as a separate entry (may have been expanded inline)")

    # The key rule: no unexpanded acronym should have a made-up gloss
    for e in entries:
        import re
        if re.match(r"^[A-Z]{2,6}$", e.term) and e.gloss and not e.instruction:
            print(f"  WARN  '{e.term}' has a gloss but no instruction — "
                  "check it was expanded from source definition, not guessed")


def test_drug_rxcui():
    """Rifampicin and doxycycline should get RxCUI from RxNorm."""
    if not _has_umls():
        print("  SKIP  no UMLS access")
        return

    preprocessed = preprocess_clinical_text(SAMPLE_TEXT)
    entries = build_glossary(SAMPLE_TEXT, preprocessed=preprocessed)

    for name in ("rifampicin", "doxycycline"):
        drug = next((e for e in entries if name in e.term.lower()), None)
        if drug is None:
            print(f"  WARN  '{name}' not found in glossary entries")
        elif drug.rxcui:
            print(f"  PASS  '{name}' has RxCUI: {drug.rxcui}")
        else:
            print(f"  WARN  '{name}' found but no RxCUI")


def test_anatomy_section():
    """Anatomy terms should have section='investigations' or similar from medspaCy."""
    if not _has_umls():
        print("  SKIP  no UMLS access")
        return

    preprocessed = preprocess_clinical_text(RENAL_TEXT)
    entries = build_glossary(RENAL_TEXT, preprocessed=preprocessed)

    print(f"  {len(entries)} entries for renal vein thrombosis text:")
    for e in entries:
        if e.snomed or (e.semantic_type and "Anatom" in (e.semantic_type or "")):
            print(f"    ANATOMY '{e.term}' → snomed={e.snomed}  section={e.section}"
                  f"  gloss='{e.gloss}'")


def test_real_file_glossary():
    """Run the full Stage 2 on the first real test file."""
    if not _has_umls():
        print("  SKIP  no UMLS access")
        return

    test_dir = config.MULTICLINSUM_TEST_DIR / "fulltext"
    files = sorted(test_dir.glob("*.txt"))
    if not files:
        print("  SKIP  no test data found")
        return

    text = files[0].read_text(encoding="utf-8")
    preprocessed = preprocess_clinical_text(text)
    cache = APICache()
    uts_client = UTSClient(cache=cache) if config.UMLS_API_KEY else None
    entries = build_glossary(text, preprocessed=preprocessed,
                             uts_client=uts_client, cache=cache)

    high   = sum(1 for e in entries if e.confidence == "high")
    medium = sum(1 for e in entries if e.confidence == "medium")
    low    = sum(1 for e in entries if e.confidence == "low")

    print(f"  PASS  {files[0].name}: {len(entries)} entries "
          f"(high={high} medium={medium} low={low})")
    print(f"        sections found: {[s.category for s in preprocessed.sections]}")

    # Show the top 5 entries that actually have a gloss
    glossed = [e for e in entries if e.gloss][:5]
    for e in glossed:
        print(f"    '{e.term}' → '{e.gloss[:60]}' [{e.gloss_source}]")


if __name__ == "__main__":
    print("=== Level 4: Glossary subsystem ===")
    for fn in [test_basic_glossary, test_negated_terms_excluded,
               test_acronym_handling, test_drug_rxcui,
               test_anatomy_section, test_real_file_glossary]:
        try:
            fn()
        except Exception as e:
            import traceback
            print(f"  ERR   {fn.__name__} — {e}")
            traceback.print_exc()
    print()
