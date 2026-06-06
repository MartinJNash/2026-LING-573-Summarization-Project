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
  "diagnosis":               "primary diagnosis exactly as stated in the report",
  "treatment_or_procedure":  ["each treatment or procedure mentioned"],
  "outcome_or_followup":     "patient outcome as stated (discharged, improved, etc.)",
  "complications_or_safety": ["complications, adverse events, or safety warnings — empty list if none"],
  "not_stated_in_source":    ["list any REQUIRED field absent from the source"],
  "presentation":            "chief complaint and how they presented",
  "exam_or_vitals":          ["key examination findings and vital signs"],
  "tests_and_findings":      ["imaging, labs, pathology results as stated"],
  "numbers_units_dates":     ["every number with its unit, e.g. '88 mmHg', '9 g Hb'"],
  "anatomy_and_laterality":  ["organ/site strings verbatim, e.g. 'left renal vein'"],
  "diagnosis_span":          "verbatim sentence(s) from the report supporting diagnosis",
  "treatment_spans":         ["verbatim phrase for each treatment"],
  "outcome_span":            "verbatim sentence(s) from the report",
  "complication_spans":      ["verbatim phrase for each complication"]
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

# Shorter retry prompt: only required fields, no verbatim span fields.
# Used when the full prompt fails (truncated JSON or missing fields).
_RETRY_SCHEMA = """\
{
  "patient_context":         "age, sex, relevant history",
  "presentation":            "chief complaint",
  "diagnosis":               "primary diagnosis as stated",
  "treatment_or_procedure":  ["each treatment or procedure"],
  "outcome_or_followup":     "patient outcome as stated, or null",
  "complications_or_safety": ["complications or adverse events, or empty list"],
  "not_stated_in_source":    ["required fields absent from the source"]
}"""

_PROMPT_RETRY = (
    "You are a clinical fact extractor. Extract ONLY the fields below from the "
    "case report. Use null or [] for any field not mentioned in the report.\n\n"
    + _RETRY_SCHEMA + "\n\n"
    "Rules: output ONLY valid JSON. No preamble, no markdown fences.\n\n"
    "CASE REPORT:\n"
)

# Truncate source for retry to avoid token overflow on very long documents
_RETRY_MAX_CHARS = 6000


def _parse_and_validate(raw: str) -> dict:
    raw = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.IGNORECASE)
    raw = re.sub(r"\s*```$", "", raw.strip())
    facts = json.loads(raw)
    missing = REQUIRED_FIELDS - set(facts.keys())
    if missing:
        raise ValueError(f"Stage 1: LLM output missing required fields: {missing}")
    return facts


def extract_facts(source_text: str, llm_fn) -> dict:
    """
    Stage 1 entry point.

    llm_fn : callable(prompt: str) -> str
    Returns a validated dict or raises ValueError.
    On JSON/field failure retries once with a shorter prompt + truncated source.
    """
    raw = llm_fn(_PROMPT_PLAIN + source_text)
    try:
        facts = _parse_and_validate(raw)
    except (json.JSONDecodeError, ValueError):
        # Retry: shorter schema, source truncated to avoid token overflow
        truncated = source_text[:_RETRY_MAX_CHARS]
        raw2 = llm_fn(_PROMPT_RETRY + truncated)
        try:
            facts = _parse_and_validate(raw2)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Stage 1: LLM returned invalid JSON: {exc}\nRaw output: {raw2[:400]}"
            )

    # Normalise list fields
    for lf in ("exam_or_vitals", "tests_and_findings", "numbers_units_dates",
               "anatomy_and_laterality", "treatment_or_procedure", "treatment_spans",
               "complications_or_safety", "complication_spans", "not_stated_in_source"):
        if facts.get(lf) is None:
            facts[lf] = []

    return facts
