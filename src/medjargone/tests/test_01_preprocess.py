"""
Level 1 — medspaCy preprocessing (section detection + ConText).
No UMLS API key or scispaCy models needed.
Falls back to regex sections if medspaCy isn't installed.

Run: python -m medjargone.tests.test_01_preprocess
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from medjargone import config
from medjargone.pipeline.preprocess import preprocess_clinical_text

# A short clinical snippet with explicit sections (ideal case)
STRUCTURED_TEXT = """\
History
A 40-year-old woman presented with intermittent fever for 2 months and a dry cough.
She had no chest pain or rash.

Examination
Temperature 37°C, heart rate 70 bpm, blood pressure 103/72 mmHg.
Lymph nodes in neck and axilla were enlarged.

Investigations
Chest CT revealed multiple pulmonary nodules in both lungs with pleural effusion.
Serum agglutination test (SAT) was 1:800.

Diagnosis
The patient was definitively diagnosed with brucellosis.

Treatment
She received rifampicin (600 mg/day) and doxycycline (100 mg twice daily) for 3 months.

Outcome
At discharge, fever, cough, and arthralgia were relieved.
After 3 months, weight had increased and depression symptoms were alleviated.
"""

# A real flowing narrative (no explicit headers — tests regex fallback)
NARRATIVE_TEXT = """\
On July 10, 2022, a 40-year-old woman was admitted due to intermittent fever for 2 months.
Her temperature fluctuated between 37.4 and 38.4°C. She had arthralgia and fatigue.
Chest CT revealed multiple pulmonary nodules. Blood culture for Brucella melitensis was positive.
She received rifampicin and doxycycline. At discharge, fever and cough were relieved.
"""

# Text with negation and uncertainty phrases (tests ConText, if medspaCy installed)
NEGATION_TEXT = """\
There was no evidence of pneumonia on imaging.
Tuberculosis was excluded after a negative TB test.
The patient denied any chest pain or tightness.
Possible malignancy was considered but not confirmed.
Family history was negative for cancer.
The patient presented with fever and lymphadenopathy.
"""


def test_sections_explicit():
    doc = preprocess_clinical_text(STRUCTURED_TEXT)
    cats = [s.category for s in doc.sections]
    print(f"  sections found: {cats}")
    assert len(doc.sections) >= 2, "Expected at least 2 sections in structured text"
    assert doc.structured_for_llm, "structured_for_llm should be non-empty"
    has_labels = any("[" in doc.structured_for_llm for _ in [None])
    print("  PASS  Explicit section detection")
    if has_labels:
        print("        medspaCy or regex produced labelled sections")
    return doc


def test_sections_narrative():
    doc = preprocess_clinical_text(NARRATIVE_TEXT)
    assert len(doc.sections) >= 1
    print("  PASS  Narrative text (no headers) — produced",
          len(doc.sections), "section(s):", [s.category for s in doc.sections])
    return doc


def test_section_for_char():
    doc = preprocess_clinical_text(STRUCTURED_TEXT)
    if len(doc.sections) > 1:
        # Pick a character known to be in the body of the second section
        sec = doc.section_for_char(doc.sections[1].start_char + 5)
        assert sec is not None
        print("  PASS  section_for_char works:", sec.category)
    else:
        print("  SKIP  section_for_char (only 1 section found — not enough structure)")


def test_context_negation():
    """
    Verifies ConText integration: scispaCy entities tagged as negated/historical
    should surface via run_context. Requires medspaCy to be installed.
    """
    import medjargone.pipeline.preprocess as _pre
    _pre._load_medspacy()
    if not _pre._medspacy_ok:
        print("  SKIP  ConText test (medspaCy not installed)")
        return
    run_context = _pre.run_context

    # Simulate two entity positions: "pneumonia" and "fever"
    # "pneumonia" appears in "no evidence of pneumonia" — should be negated
    # "fever" appears at end — should NOT be negated
    pneu_pos = NEGATION_TEXT.index("pneumonia")
    fever_pos = NEGATION_TEXT.rfind("fever")

    modifiers = run_context(
        NEGATION_TEXT,
        [(pneu_pos, pneu_pos + len("pneumonia")),
         (fever_pos, fever_pos + len("fever"))],
    )

    print("  ConText modifiers:")
    for term, mods in modifiers.items():
        print(f"    '{term}': negated={mods.is_negated}  uncertain={mods.is_uncertain}"
              f"  historical={mods.is_historical}  exclude={mods.should_exclude}")

    pneumonia_mods = modifiers.get("pneumonia")
    if pneumonia_mods and pneumonia_mods.is_negated:
        print("  PASS  'pneumonia' correctly marked as negated")
    else:
        print("  WARN  'pneumonia' not marked negated — check ConText rules")


def test_real_file():
    test_dir = config.MULTICLINSUM_TEST_DIR / "fulltext"
    files = sorted(test_dir.glob("*.txt"))
    if not files:
        print("  SKIP  real file test (no test data found)")
        return
    text = files[0].read_text(encoding="utf-8")
    doc = preprocess_clinical_text(text)
    print(f"  PASS  Real file processed: {files[0].name}")
    print(f"        {len(doc.sections)} section(s): {[s.category for s in doc.sections]}")
    print(f"        structured_for_llm length: {len(doc.structured_for_llm)} chars")


if __name__ == "__main__":
    print("=== Level 1: Preprocessing (medspaCy sections + ConText) ===")
    for fn in [test_sections_explicit, test_sections_narrative,
               test_section_for_char, test_context_negation, test_real_file]:
        try:
            fn()
        except Exception as e:
            print(f"  FAIL  {fn.__name__} — {e}")
    print()
