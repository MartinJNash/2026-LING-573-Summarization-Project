"""
Level 5 — Full pipeline (Stages 1–4) with a mock LLM.
No vllm, no GPU needed. Wires the entire pipeline with a scripted LLM response
so you can verify all stages connect correctly.

Run: python -m medjargone.tests.test_05_pipeline_mock
"""

import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from medjargone import config
from medjargone.pipeline.preprocess import preprocess_clinical_text
from medjargone.pipeline.extract_facts import extract_facts
from medjargone.pipeline.rewrite import generate_rewrite
from medjargone.pipeline.verify import verify_and_fix, rule_check
from medjargone.pipeline.glossary import build_glossary, APICache

SOURCE_TEXT = """\
A 43-year-old woman at 30 days post-partum presented with left flank pain and hematuria \
for one week. Blood pressure was 158/67 mmHg, heart rate was 90 bpm.
Hemoglobin was 9 g. CT urography showed an ectatic left renal vein with a large hypodense \
thrombus extending to the inferior vena cava, causing renal venous infarction.
Anticoagulation with heparin was started. The patient was admitted and stabilised.
"""

# Scripted fact extraction response — what the LLM would ideally return
_SCRIPTED_FACTS = {
    "patient_context": "43-year-old woman, 30 days post-partum",
    "presentation": "left flank pain and hematuria for one week",
    "exam_or_vitals": ["blood pressure 158/67 mmHg", "heart rate 90 bpm"],
    "tests_and_findings": [
        "CT showed ectatic left renal vein with large hypodense thrombus",
        "hemoglobin 9 g",
    ],
    "numbers_units_dates": ["158/67 mmHg", "90 bpm", "9 g"],
    "anatomy_and_laterality": ["left renal vein", "inferior vena cava"],
    "diagnosis": "renal venous infarction",
    "diagnosis_span": "CT showed an ectatic left renal vein with a large hypodense thrombus causing renal venous infarction",
    "treatment_or_procedure": ["anticoagulation with heparin"],
    "treatment_spans": ["Anticoagulation with heparin was started"],
    "outcome_or_followup": "admitted and stabilised",
    "outcome_span": "The patient was admitted and stabilised",
    "complications_or_safety": [],
    "complication_spans": [],
    "not_stated_in_source": [],
}

_SCRIPTED_SUMMARY = (
    "A 43-year-old woman who had recently given birth was admitted with pain on the left side "
    "and blood in the urine for one week. Her blood pressure was 158/67 mmHg and heart rate "
    "was 90 bpm. Her blood count showed low hemoglobin (9 g). "
    "A scan found a large blood clot in the left renal vein extending to the inferior vena cava, "
    "causing damage to the left kidney tissue (renal venous infarction). "
    "She was given a blood-thinning medication (heparin) and was admitted and stabilised."
)


def make_mock_llm(stage_responses: dict) -> object:
    """
    Returns a mock llm_fn that returns scripted responses for each stage.
    Detects which stage is calling by inspecting the prompt prefix.
    """
    call_log = []

    def llm_fn(prompt: str) -> str:
        # Stage 1 prompts ask for JSON extraction
        if "fact extractor" in prompt.lower() or "CASE REPORT" in prompt:
            call_log.append("stage1")
            return json.dumps(stage_responses.get("s1", _SCRIPTED_FACTS))
        # Stage 3 prompts ask for a patient summary
        elif "patient-friendly summary" in prompt.lower() or "FACTS:" in prompt:
            call_log.append("stage3")
            return stage_responses.get("s3", _SCRIPTED_SUMMARY)
        # Stage 4 fix prompts mention "issues"
        elif "ISSUES:" in prompt or "Fix ONLY these issues" in prompt:
            call_log.append("stage4_fix")
            return stage_responses.get("s4_fix", prompt.split("ORIGINAL SUMMARY:")[-1].strip())
        else:
            call_log.append("unknown")
            return ""

    llm_fn.call_log = call_log
    return llm_fn


def test_extract_facts_mock():
    llm = make_mock_llm({})
    preprocessed = preprocess_clinical_text(SOURCE_TEXT)
    facts = extract_facts(SOURCE_TEXT, llm, preprocessed=preprocessed)
    assert facts["diagnosis"] == "renal venous infarction"
    assert "left renal vein" in facts["anatomy_and_laterality"]
    assert facts["treatment_or_procedure"]
    print("  PASS  Stage 1 extract_facts (mock LLM)")
    print(f"        sections_used={facts.get('_sections_used')}")
    return facts


def test_rewrite_mock(facts=None):
    llm = make_mock_llm({})
    if facts is None:
        facts = _SCRIPTED_FACTS
    preprocessed = preprocess_clinical_text(SOURCE_TEXT)
    entries = build_glossary(SOURCE_TEXT, preprocessed=preprocessed) if config.UMLS_API_KEY or config.UMLS_SUBSET_DB.exists() else []
    summary = generate_rewrite(facts, entries, llm)
    assert summary.strip()
    print(f"  PASS  Stage 3 generate_rewrite (mock LLM) — {len(summary)} chars")
    print(f"        preview: {summary[:120]}…")
    return summary


def test_verify_good_summary():
    final, report = verify_and_fix(
        _SCRIPTED_SUMMARY, _SCRIPTED_FACTS, SOURCE_TEXT,
        make_mock_llm({})
    )
    print(f"  verify result: {report.summary_str()}")
    # The scripted summary should pass Tier-1
    assert not report.missing_coverage, f"Unexpected missing coverage: {report.missing_coverage}"
    print("  PASS  Good summary passes Tier-1 verify_and_fix")
    return report


def test_verify_bad_summary():
    """A summary missing numbers, laterality, and diagnosis should fail Tier-1."""
    bad_summary = (
        "A woman was admitted with flank pain. She had a kidney problem. "
        "She was given medication and got better."
    )
    final, report = verify_and_fix(
        bad_summary, _SCRIPTED_FACTS, SOURCE_TEXT,
        make_mock_llm({"s4_fix": _SCRIPTED_SUMMARY})
    )
    print(f"  Tier-1 failures detected: {report.summary_str()}")
    assert report.has_rule_failures, "Expected Tier-1 failures for bad summary"
    assert report.revised, "Expected re-prompt to have been issued"
    print("  PASS  Bad summary triggers re-prompt and is replaced")


def test_full_pipeline_mock():
    """Run all 4 stages together."""
    llm = make_mock_llm({})

    preprocessed = preprocess_clinical_text(SOURCE_TEXT)
    facts = extract_facts(SOURCE_TEXT, llm, preprocessed=preprocessed)

    cache = APICache()
    entries = build_glossary(
        SOURCE_TEXT, preprocessed=preprocessed, cache=cache
    )

    summary = generate_rewrite(facts, entries, llm)
    final, report = verify_and_fix(summary, facts, SOURCE_TEXT, llm, entries)

    print(f"  PASS  Full pipeline (mock LLM):")
    print(f"        facts: diagnosis='{facts['diagnosis']}'")
    print(f"        glossary: {len(entries)} entries")
    print(f"        summary: {len(final)} chars")
    print(f"        verification: {report.summary_str()}")
    print(f"        LLM calls: {llm.call_log}")


def test_stage1_invalid_json():
    """Stage 1 must raise ValueError, not crash, on bad LLM output."""
    def bad_llm(prompt): return "This is not JSON at all."
    try:
        extract_facts(SOURCE_TEXT, bad_llm)
        print("  FAIL  Should have raised ValueError")
    except ValueError as e:
        print(f"  PASS  Invalid JSON raises ValueError: {str(e)[:80]}")


def test_stage1_missing_fields():
    """Stage 1 must raise ValueError if required fields are absent."""
    def partial_llm(prompt):
        return json.dumps({"diagnosis": "something", "patient_context": "old"})
    try:
        extract_facts(SOURCE_TEXT, partial_llm)
        print("  FAIL  Should have raised ValueError for missing fields")
    except ValueError as e:
        print(f"  PASS  Missing required fields raises ValueError: {str(e)[:80]}")


if __name__ == "__main__":
    print("=== Level 5: Full pipeline with mock LLM (no GPU needed) ===")
    for fn in [test_stage1_invalid_json, test_stage1_missing_fields,
               test_extract_facts_mock, test_rewrite_mock,
               test_verify_good_summary, test_verify_bad_summary,
               test_full_pipeline_mock]:
        try:
            fn()
        except Exception as e:
            import traceback
            print(f"  ERR   {fn.__name__} — {e}")
            traceback.print_exc()
    print()
