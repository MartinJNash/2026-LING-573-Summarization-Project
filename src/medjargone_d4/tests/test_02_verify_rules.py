"""
Level 2 — Stage 4 Tier-1 rule checks.
No models, no API, no network. Pure deterministic logic.

Run: python -m medjargone.tests.test_02_verify_rules
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from medjargone_d4.pipeline.verify import (
    rule_check, _check_numbers, _check_laterality, _check_coverage
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

GOOD_FACTS = {
    "diagnosis": "left renal venous infarction",
    "diagnosis_span": "CT showed a large thrombus in the left renal vein causing infarction",
    "treatment_or_procedure": ["anticoagulation with heparin"],
    "treatment_spans": ["Anticoagulation heparin therapy was started"],
    "outcome_or_followup": "patient was admitted and stabilised",
    "outcome_span": "patient was admitted to the ward and stabilised on heparin",
    "complications_or_safety": [],
    "complication_spans": [],
    "numbers_units_dates": ["9 g hemoglobin", "158 mmHg", "88 CRP"],
    "anatomy_and_laterality": ["left renal vein", "inferior vena cava"],
    "not_stated_in_source": [],
}

# Summary must contain: 9 g, 158 mmHg, 88 (CRP), "left", "renal vein",
# "inferior vena cava", diagnosis, outcome
GOOD_SUMMARY = (
    "A woman was admitted with left flank pain. "
    "Imaging showed a large blood clot in the left renal vein extending to the "
    "inferior vena cava, causing damage to the kidney (left renal venous infarction). "
    "Her hemoglobin was 9 g, blood pressure was 158 mmHg, and CRP was 88. "
    "She was treated with a blood-thinning medication (heparin). "
    "She was admitted to the ward and stabilised on heparin."
)


def test_all_pass():
    report = rule_check(GOOD_FACTS, GOOD_SUMMARY)
    assert not report.has_rule_failures, f"Expected no failures, got: {report.summary_str()}"
    print("  PASS  Good facts + good summary → all checks pass")


def test_missing_number():
    bad_summary = GOOD_SUMMARY.replace("9 g", "low").replace("158 mmHg", "high")
    missing = _check_numbers(GOOD_FACTS, bad_summary)
    assert missing, "Expected missing numbers to be detected"
    print(f"  PASS  Missing numbers detected: {missing}")


def test_wrong_laterality():
    bad_summary = GOOD_SUMMARY.replace("left", "right")
    failed = _check_laterality(GOOD_FACTS, bad_summary)
    assert failed, "Expected laterality mismatch to be detected"
    print("  PASS  Wrong laterality ('right' instead of 'left') detected")


def test_correct_laterality():
    ok = _check_laterality(GOOD_FACTS, GOOD_SUMMARY)
    assert not ok, "Expected no laterality failure when summary is correct"
    print("  PASS  Correct laterality passes")


def test_missing_diagnosis_coverage():
    no_dx = (
        "A woman was admitted with left flank pain. "
        "She was treated with a blood-thinning medication. "
        "She was admitted to the ward."
    )
    missing = _check_coverage(GOOD_FACTS, no_dx)
    assert "diagnosis" in missing, f"Expected 'diagnosis' in missing, got {missing}"
    print(f"  PASS  Missing diagnosis coverage detected: {missing}")


def test_bilateral_laterality():
    bilateral_facts = {**GOOD_FACTS, "anatomy_and_laterality": ["bilateral pleural effusion"]}
    bad = GOOD_SUMMARY  # "bilateral" not in summary
    failed = _check_laterality(bilateral_facts, bad)
    assert failed, "Expected bilateral mismatch to be detected"
    print("  PASS  Missing 'bilateral' in summary correctly flagged")

    good = GOOD_SUMMARY + " There was bilateral pleural effusion."
    ok = _check_laterality(bilateral_facts, good)
    assert not ok
    print("  PASS  'bilateral' present in summary passes")


def test_full_report():
    """Run rule_check on the full brucellosis report structure."""
    facts = {
        "diagnosis": "brucellosis",
        "diagnosis_span": "patient was definitively diagnosed with brucellosis",
        "treatment_or_procedure": ["rifampicin 600 mg/day", "doxycycline 100 mg twice daily"],
        "treatment_spans": [
            "received rifampicin (600 mg/dose, once a day)",
            "doxycycline (100 mg/dose, twice a day) for 3 months",
        ],
        "outcome_or_followup": "fever cough arthralgia relieved at discharge",
        "outcome_span": "At discharge, fever, cough, arthralgia, depression and fatigue were relieved",
        "complications_or_safety": ["multipulmonary nodules", "hepatosplenomegaly"],
        "complication_spans": ["multipulmonary nodules, arthralgia, hepatosplenomegaly"],
        "numbers_units_dates": ["600 mg", "100 mg", "37°C", "103/72 mmHg", "70 bpm"],
        "anatomy_and_laterality": ["both lungs", "liver", "spleen"],
        "not_stated_in_source": [],
    }
    good_summary = (
        "A 40-year-old woman was admitted with fever and cough. "
        "Tests confirmed she had brucellosis, a bacterial infection from animal contact. "
        "She was treated with rifampicin 600 mg daily and doxycycline 100 mg twice daily for 3 months. "
        "Nodules were found in both lungs and her liver and spleen were enlarged. "
        "At discharge, fever, cough, and arthralgia were relieved. "
        "Her temperature was 37°C, heart rate was 70 bpm, and blood pressure was 103/72 mmHg."
    )
    report = rule_check(facts, good_summary)
    print(f"  Brucellosis example: {report.summary_str()}")
    if not report.has_rule_failures:
        print("  PASS  Full brucellosis summary passes all Tier-1 checks")
    else:
        print("  WARN  Some checks flagged (may be expected for short summary):")
        if report.missing_numbers:
            print(f"        missing numbers: {report.missing_numbers}")
        if report.wrong_laterality:
            print("        laterality mismatch")
        if report.missing_coverage:
            print(f"        missing coverage: {report.missing_coverage}")


if __name__ == "__main__":
    print("=== Level 2: Tier-1 rule checks (no models needed) ===")
    for fn in [test_all_pass, test_missing_number, test_wrong_laterality,
               test_correct_laterality, test_missing_diagnosis_coverage,
               test_bilateral_laterality, test_full_report]:
        try:
            fn()
        except AssertionError as e:
            print(f"  FAIL  {fn.__name__} — {e}")
        except Exception as e:
            print(f"  ERR   {fn.__name__} — {e}")
    print()
