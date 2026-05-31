"""
Stage 1 — Whole-document clinical fact extraction.

Input : source text (str)
Output: validated fact schema dict (every field carries source spans)

Processes the full document in one pass — avoids the D2/D3 positional truncation
failure where only the opening sentences were summarised.

Mandatory fields: diagnosis, treatment_or_procedure, outcome_or_followup,
complications_or_safety. Absent fields go to not_stated_in_source — they are
never silently dropped.
"""

from __future__ import annotations

import json
import re

REQUIRED_FIELDS = {
    "patient_context",
    "presentation",
    "diagnosis",
    "treatment_or_procedure",
    "outcome_or_followup",
    "complications_or_safety",
}

_SCHEMA_BLOCK = """\
{
  "patient_context":         "age, sex, relevant history — no name, no address",
  "presentation":            "chief complaint and how they presented",
  "exam_or_vitals":          ["key examination findings and vital signs"],
  "tests_and_findings":      ["imaging, labs, pathology results as stated"],
  "numbers_units_dates":     ["every number with its unit, e.g. '88 mmHg', '9 g Hb'"],
  "anatomy_and_laterality":  ["organ/site strings verbatim, e.g. 'left renal vein'"],
  "diagnosis":               "primary diagnosis exactly as stated in the report",
  "diagnosis_span":          "verbatim sentence(s) from the report supporting diagnosis",
  "treatment_or_procedure":  ["each treatment or procedure mentioned"],
  "treatment_spans":         ["verbatim phrase for each treatment"],
  "outcome_or_followup":     "patient outcome as stated (discharged, improved, etc.)",
  "outcome_span":            "verbatim sentence(s) from the report",
  "complications_or_safety": ["complications, adverse events, or safety warnings — empty list if none"],
  "complication_spans":      ["verbatim phrase for each complication"],
  "not_stated_in_source":    ["list any REQUIRED field that is absent from the source"]
}"""

_RULES = """\
Rules:
- Every *_span field must quote text VERBATIM from the source.
- If a required field has no data write null or [] AND add the field name to
  not_stated_in_source. Never invent or infer.
- Output ONLY valid JSON. No preamble, no markdown fences, no comments."""

_PROMPT_PLAIN = (
    "You are a clinical fact extractor. Read the case report below and return "
    "a JSON object with EXACTLY the following fields. Do not add extra fields.\n\n"
    + _SCHEMA_BLOCK + "\n\n"
    + _RULES
    + "\n\nCASE REPORT:\n"
)

def extract_facts(source_text: str, llm_fn) -> dict:
    """
    Stage 1 entry point.

    llm_fn : callable(prompt: str) -> str
    Returns a validated dict or raises ValueError.
    """
    raw = llm_fn(_PROMPT_PLAIN + source_text)

    # Strip markdown fences
    raw = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.IGNORECASE)
    raw = re.sub(r"\s*```$", "", raw.strip())

    try:
        facts = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Stage 1: LLM returned invalid JSON: {exc}\nRaw output: {raw[:400]}"
        )

    missing = REQUIRED_FIELDS - set(facts.keys())
    if missing:
        raise ValueError(
            f"Stage 1: LLM output missing required fields: {missing}"
        )

    # Normalise list fields
    for lf in ("exam_or_vitals", "tests_and_findings", "numbers_units_dates",
               "anatomy_and_laterality", "treatment_or_procedure", "treatment_spans",
               "complications_or_safety", "complication_spans", "not_stated_in_source"):
        if facts.get(lf) is None:
            facts[lf] = []

    return facts
