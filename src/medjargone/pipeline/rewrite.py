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

from medjargone.pipeline.glossary import GlossaryEntry


_REWRITE_PROMPT = """\
Write a patient-friendly summary using ONLY the verified facts and glossary below.
Add no diagnosis, cause, outcome, reassurance, or mechanism not in the facts.
Replace medical terms with their glossary plain-language version; if a term has no
gloss, keep it with a short neutral description of its type (e.g. "a medication").
Order:
  (1) why they came in
  (2) what doctors found
  (3) main problem / diagnosis
  (4) treatment or procedure
  (5) outcome / follow-up
  (6) any complication or warning

Keep all numbers, dates, anatomy, and left/right exactly as given in the facts.
Write in plain prose at a Grade 7 reading level. No headings.
Output ONLY the summary text.

FACTS:
{facts_json}

GLOSSARY:
{glossary_json}
"""


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
    active = [
        {
            "term": e.term,
            "semantic_type": e.semantic_type,
            "gloss": e.gloss,
            "confidence": e.confidence,
            **({"instruction": e.instruction} if e.instruction else {}),
        }
        for e in glossary
        if e.needs_gloss
    ]

    prompt = _REWRITE_PROMPT.format(
        facts_json=json.dumps(facts, indent=2),
        glossary_json=json.dumps(active, indent=2),
    )

    return llm_fn(prompt).strip()
