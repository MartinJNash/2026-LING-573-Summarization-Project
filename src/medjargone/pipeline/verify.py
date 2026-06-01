"""
Stage 4 — 2-tier verify-and-fix.

Input : draft summary (str), facts dict (Stage 1), source text (str)
Output: (final_summary: str, VerificationReport)

Tier 1 — Deterministic rules (no model):
  • Numbers / units / dates
  • Laterality (left / right / bilateral)
  • Organ / site identity  — string match against anatomy_and_laterality facts
  • Coverage               — mandatory schema fields present if stated in source

Tier 2 — MiniCheck claim-level faithfulness:
  Decomposes the summary into atomic claims and checks each against the
  relevant source span (falling back to the full source text).
  Claims below tau_low are flagged as unsupported.

Any Tier-1 failure or Tier-2 fail → one targeted re-prompt.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from medjargone import config
from medjargone.pipeline.glossary import GlossaryEntry

# Lazy model handle — _minicheck_tried prevents repeated import attempts / warnings
_minicheck       = None
_minicheck_tried = False


# ── Verification report ────────────────────────────────────────────────────────

@dataclass
class VerificationReport:
    # Tier 1
    missing_numbers: list[str]    = field(default_factory=list)
    wrong_laterality: bool        = False
    wrong_organs: list[str]       = field(default_factory=list)
    missing_coverage: list[str]   = field(default_factory=list)
    # Tier 2
    unsupported_claims: list[str] = field(default_factory=list)
    gray_zone_claims: list[str]   = field(default_factory=list)
    # Housekeeping
    revised: bool                 = False

    @property
    def has_rule_failures(self) -> bool:
        return bool(
            self.missing_numbers
            or self.wrong_laterality
            or self.missing_coverage
        )

    @property
    def has_model_failures(self) -> bool:
        return bool(self.unsupported_claims)

    def summary_str(self) -> str:
        parts = []
        if self.missing_numbers:
            parts.append(f"missing numbers: {self.missing_numbers}")
        if self.wrong_laterality:
            parts.append("laterality mismatch")
        if self.wrong_organs:
            parts.append(f"organ mismatch: {self.wrong_organs}")
        if self.missing_coverage:
            parts.append(f"missing coverage: {self.missing_coverage}")
        if self.unsupported_claims:
            parts.append(f"{len(self.unsupported_claims)} unsupported claim(s)")
        if self.gray_zone_claims:
            parts.append(f"{len(self.gray_zone_claims)} gray-zone claim(s)")
        return "; ".join(parts) if parts else "all checks passed"


# ── Tier 1: Deterministic rules ───────────────────────────────────────────────

_NUM_RE = re.compile(
    r"\b\d+(?:[.,]\d+)?\s*(?:mg|ml|g|mmol|mmhg|%|bpm|kg|cm|mm|iu|meq)\b",
    re.IGNORECASE,
)


def _as_str(val) -> str:
    """Coerce a fact field to str — model sometimes returns lists for string fields."""
    if val is None:
        return ""
    if isinstance(val, list):
        return " ".join(str(v) for v in val)
    return str(val)


def _nums(text: str) -> set[str]:
    return {m.group().lower().replace(",", ".").replace(" ", "")
            for m in _NUM_RE.finditer(text)}


def _check_numbers(facts: dict, summary: str) -> list[str]:
    source_nums: set[str] = set()
    for key in ("diagnosis_span", "outcome_span"):
        source_nums |= _nums(_as_str(facts.get(key)))
    for spans_key in ("treatment_spans", "complication_spans"):
        for s in facts.get(spans_key) or []:
            source_nums |= _nums(_as_str(s))
    for n in facts.get("numbers_units_dates") or []:
        source_nums |= _nums(_as_str(n))
    return sorted(source_nums - _nums(summary))


def _check_laterality(facts: dict, summary: str) -> bool:
    lat_terms = " ".join(facts.get("anatomy_and_laterality") or []).lower()
    if not lat_terms:
        return False
    summary_lc = summary.lower()
    for word in ("left", "right", "bilateral"):
        if word in lat_terms and word not in summary_lc:
            return True
    return False


def _check_coverage(facts: dict, summary: str) -> list[str]:
    summary_lc = summary.lower()
    missing = []
    for field_name, label in [
        ("diagnosis", "diagnosis"),
        ("outcome_or_followup", "outcome"),
    ]:
        val = _as_str(facts.get(field_name))
        if val in (facts.get("not_stated_in_source") or []):
            continue
        words = [w for w in re.findall(r"\b\w{5,}\b", val.lower())
                 if w not in {"which", "there", "these", "their", "where",
                              "about", "after", "since", "while"}]
        if words and not any(w in summary_lc for w in words[:3]):
            missing.append(label)
    return missing


def rule_check(facts: dict, summary: str) -> VerificationReport:
    """Public entry point for Tier-1 deterministic checks only (no models)."""
    report = VerificationReport()
    report.missing_numbers  = _check_numbers(facts, summary)
    report.wrong_laterality = _check_laterality(facts, summary)
    report.missing_coverage = _check_coverage(facts, summary)
    report.wrong_organs     = _check_organ_identity(facts, summary)
    return report


def _check_organ_identity(facts: dict, summary: str) -> list[str]:
    """String-match anatomy_and_laterality terms from the facts against the summary."""
    source_anatomy = [a.lower() for a in (facts.get("anatomy_and_laterality") or [])]
    summary_lc = summary.lower()
    return [t for t in source_anatomy if t not in summary_lc]


# ── Tier 2: MiniCheck ─────────────────────────────────────────────────────────

def _load_minicheck():
    global _minicheck, _minicheck_tried
    if _minicheck_tried:
        return
    _minicheck_tried = True
    try:
        from minicheck.minicheck import MiniCheck
        _minicheck = MiniCheck(model_name=config.MINICHECK_MODEL, enable_prefix_caching=False)
    except ImportError:
        print("[warn] MiniCheck not installed — Tier 2 skipped (once)")
    except Exception as exc:
        print(f"[warn] MiniCheck load failed: {exc}")


def _split_claims(summary: str) -> list[str]:
    """Split summary into atomic claims (sentence level)."""
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", summary)
            if len(s.strip()) > 15]


def _best_premise(claim: str, facts: dict, source_text: str) -> str:
    """
    Choose the most focused premise for a claim:
    return the first span field whose words overlap with the claim,
    falling back to the full source text.
    """
    claim_words = set(re.findall(r"\b\w{4,}\b", claim.lower()))
    for key in ("diagnosis_span", "outcome_span"):
        span = _as_str(facts.get(key))
        if span and len(claim_words & set(re.findall(r"\b\w{4,}\b", span.lower()))) >= 2:
            return span
    for spans_key in ("treatment_spans", "complication_spans"):
        for span in (facts.get(spans_key) or []):
            span = _as_str(span)
            if span and len(claim_words & set(re.findall(r"\b\w{4,}\b", span.lower()))) >= 2:
                return span
    return source_text


def minicheck_verify(
    summary: str,
    facts: dict,
    source_text: str,
) -> tuple[list[str], list[str]]:
    """
    Returns (unsupported_claims, gray_zone_claims).
    """
    _load_minicheck()
    if _minicheck is None:
        return [], []

    claims = _split_claims(summary)
    unsupported = []
    gray_zone   = []

    premises = [_best_premise(c, facts, source_text) for c in claims]

    try:
        _, scores = _minicheck.score(docs=premises, claims=claims)
    except Exception as exc:
        print(f"[warn] MiniCheck scoring failed: {exc}")
        return [], []

    for claim, score in zip(claims, scores):
        if score < config.MINICHECK_TAU_LOW:
            unsupported.append(claim)
        elif score < config.MINICHECK_TAU_HIGH:
            gray_zone.append(claim)

    return unsupported, gray_zone


# ── Fix prompt ────────────────────────────────────────────────────────────────

_FIX_PROMPT = """\
The patient summary below has the following specific issues. Fix ONLY these issues \
and return the corrected summary. Do not change anything else.
Do NOT explain what you changed. Do NOT add headings or preamble.
Begin the corrected summary immediately with the first sentence about the patient.

ISSUES:
{issues}

ORIGINAL SUMMARY:
{summary}

CORRECTED SUMMARY:"""


def _build_fix_text(report: VerificationReport, unsupported: list[str]) -> str:
    lines = []
    if report.missing_numbers:
        lines.append(
            "- These numbers must appear exactly as given: "
            + ", ".join(report.missing_numbers)
        )
    if report.wrong_laterality:
        lines.append(
            "- The left/right/bilateral designation from the facts is wrong or missing."
        )
    if report.missing_coverage:
        lines.append(
            "- Required information is missing from the summary: "
            + ", ".join(report.missing_coverage)
        )
    for claim in unsupported:
        lines.append(
            f"- This claim is NOT supported by the source report and must be removed "
            f"or corrected: \"{claim}\""
        )
    return "\n".join(lines)


# ── Public API ────────────────────────────────────────────────────────────────

def verify_and_fix(
    draft: str,
    facts: dict,
    source_text: str,
    llm_fn,
) -> tuple[str, VerificationReport]:
    """
    Stage 4 entry point.

    llm_fn : callable(prompt: str) -> str
    Returns (final_summary, report).
    """
    report = VerificationReport()

    # Tier 1 — deterministic rules
    report.missing_numbers  = _check_numbers(facts, draft)
    report.wrong_laterality = _check_laterality(facts, draft)
    report.missing_coverage = _check_coverage(facts, draft)
    report.wrong_organs     = _check_organ_identity(facts, draft)

    # Tier 2 — MiniCheck
    unsupported, gray_zone = minicheck_verify(draft, facts, source_text)
    report.unsupported_claims = unsupported
    report.gray_zone_claims   = gray_zone

    final = draft
    if report.has_rule_failures or report.has_model_failures:
        fix_text = _build_fix_text(report, report.unsupported_claims)
        revised  = llm_fn(
            _FIX_PROMPT.format(issues=fix_text, summary=draft),
            max_tokens=config.LLM_MAX_NEW_TOKENS_S3,
        ).strip()
        if revised:
            final = revised
            report.revised = True

    return final, report
