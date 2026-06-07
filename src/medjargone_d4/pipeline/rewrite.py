"""
Stage 3 — Fact-conditioned patient-friendly rewrite.

Input : facts dict (Stage 1), glossary entries (Stage 2)
Output: patient-friendly summary string

The LLM sees only the extracted facts and the grounded UMLS-backed glossary.
It cannot invent information because the prompt explicitly prohibits anything
not present in the facts schema.
"""

from __future__ import annotations

import json

from medjargone_d4 import config
from medjargone_d4.pipeline.glossary import GlossaryEntry


_REWRITE_PROMPT = """\
Write a patient-friendly summary using ONLY the verified facts and glossary below.
Add no diagnosis, cause, outcome, reassurance, or mechanism not in the facts.
Replace medical terms with their plain-language version from the glossary; if a term \
has no gloss, keep it with a short neutral description (e.g. "a medication").

Begin with the patient's age and sex if known. Cover in 4-5 sentences:
  (1) why they came in
  (2) main problem / diagnosis
  (3) treatment or procedure
  (4) outcome / follow-up and any complications

Aim for 100-150 words. Do NOT list every test result — include only the key finding \
that led to the diagnosis.
Keep all numbers with units, dates, anatomy, and left/right exactly as given in the facts.
Write in plain prose at a Grade 7 reading level. No headings. No preamble.
Do NOT explain what you are doing. Do NOT repeat these instructions.
Begin the summary immediately with the first sentence about the patient.

FACTS:
{facts_json}

GLOSSARY:
{glossary_json}

SUMMARY:"""


_CONFIDENCE_ORDER = {"high": 0, "medium": 1, "low": 2}
_MAX_GLOSSARY_ENTRIES = 20


def generate_rewrite(
    facts: dict,
    glossary: list[GlossaryEntry],
    llm_fn,
) -> str:
    """
    Stage 3 entry point.

    llm_fn : callable(prompt: str) -> str
    Returns the rewritten summary string.
    """
    top_entries = sorted(
        [e for e in glossary if e.needs_gloss],
        key=lambda e: _CONFIDENCE_ORDER.get(e.confidence, 3),
    )[:_MAX_GLOSSARY_ENTRIES]

    active = [
        {
            "term": e.term,
            "semantic_type": e.semantic_type,
            "gloss": e.gloss,
            "confidence": e.confidence,
            **({"instruction": e.instruction} if e.instruction else {}),
        }
        for e in top_entries
    ]

    prompt = _REWRITE_PROMPT.format(
        facts_json=json.dumps(facts, indent=2),
        glossary_json=json.dumps(active, indent=2),
    )

    return llm_fn(prompt, max_tokens=config.LLM_MAX_NEW_TOKENS_S3).strip()
