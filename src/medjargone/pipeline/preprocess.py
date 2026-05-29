"""
Clinical text preprocessing.

Responsibilities:
  1. Section detection — splits the report into named clinical zones
     (history, examination, diagnosis, treatment, outcome, …)
     Uses regex-based detection tuned for case report headings.
  2. ConText modifiers — tags each entity span as negated / uncertain /
     historical / hypothetical / family (so the glossary can filter them)
     Uses medspaCy ConText if installed; otherwise no filtering.
  3. Structured text for Stage 1 — reformats the source with explicit
     section headers so the LLM's extraction prompt is better anchored

medspaCy is an optional dependency used only for ConText negation/modifier
detection. Section detection always uses the regex approach, which is more
reliable for case reports than medspaCy's clinical-note-oriented sectionizer.

Install:
    pip install medspacy
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from medjargone import config

# ── Lazy model handle ─────────────────────────────────────────────────────────
_med_nlp = None
_medspacy_ok = False


# ── Data types ─────────────────────────────────────────────────────────────────

@dataclass
class ClinicalSection:
    category: str       # e.g. "history", "diagnosis", "treatment", "outcome"
    title: str          # heading text as it appeared; "narrative" if implicit
    body: str           # section body text
    start_char: int
    end_char: int


@dataclass
class EntityModifiers:
    is_negated:     bool = False
    is_uncertain:   bool = False
    is_historical:  bool = False
    is_hypothetical: bool = False
    is_family:      bool = False

    @property
    def should_exclude(self) -> bool:
        """True when the entity should be dropped from the glossary."""
        return (self.is_negated or self.is_historical
                or self.is_hypothetical or self.is_family)


@dataclass
class PreprocessedDoc:
    text: str
    sections: list[ClinicalSection]
    # lowercased span text → ConText modifiers (populated after ConText is run)
    entity_modifiers: dict[str, EntityModifiers] = field(default_factory=dict)
    structured_for_llm: str = ""        # section-tagged text for Stage 1

    def section_for_char(self, char_pos: int) -> Optional[ClinicalSection]:
        """Return the ClinicalSection containing char_pos, or None."""
        for sec in self.sections:
            if sec.start_char <= char_pos < sec.end_char:
                return sec
        return None


# ── Section rules for clinical case reports ───────────────────────────────────
# (category, list-of-heading-literals that signal the start of that section)
_SECTION_PATTERNS: list[tuple[str, list[str]]] = [
    ("patient_context",   ["demographics", "background", "patient information"]),
    ("history",           ["history", "history of present illness", "hpi",
                           "presenting complaint", "chief complaint",
                           "case report", "case presentation", "case description",
                           "clinical history", "clinical presentation", "medical history"]),
    ("examination",       ["examination", "physical examination", "physical exam",
                           "clinical examination", "on examination", "vitals",
                           "vital signs"]),
    ("investigations",    ["investigations", "laboratory", "labs", "imaging",
                           "radiology", "pathology", "tests", "results",
                           "diagnostic", "workup", "laboratory results"]),
    ("diagnosis",         ["diagnosis", "assessment", "impression",
                           "differential diagnosis", "final diagnosis"]),
    ("treatment",         ["treatment", "management", "therapy", "intervention",
                           "procedure", "surgery", "operation", "plan"]),
    ("outcome",           ["outcome", "follow-up", "followup", "follow up",
                           "discharge", "progress", "course", "recovery",
                           "result", "post-operative", "postoperative"]),
    ("complications",     ["complications", "adverse events", "safety"]),
    ("discussion",        ["discussion"]),
    ("conclusion",        ["conclusion", "conclusions", "summary"]),
]

# Map section categories to their user-facing header labels
_SECTION_LABELS: dict[str, str] = {
    "patient_context":  "PATIENT BACKGROUND",
    "history":          "HISTORY & PRESENTATION",
    "examination":      "EXAMINATION & VITALS",
    "investigations":   "INVESTIGATIONS & FINDINGS",
    "diagnosis":        "DIAGNOSIS",
    "treatment":        "TREATMENT & PROCEDURES",
    "outcome":          "OUTCOME & FOLLOW-UP",
    "complications":    "COMPLICATIONS",
    "discussion":       "DISCUSSION",
    "conclusion":       "CONCLUSION",
    "narrative":        "FULL REPORT",
}


# ── medspaCy loading (ConText only) ──────────────────────────────────────────

def _load_medspacy():
    """Load medspaCy with sentence splitter + ConText for negation/modifier detection."""
    global _med_nlp, _medspacy_ok
    if _med_nlp is not None:
        return

    try:
        import logging
        import medspacy

        # Suppress PyRuSH debug token traces
        logging.getLogger("PyRuSH").setLevel(logging.WARNING)

        try:
            _med_nlp = medspacy.load(medspacy_enable=[])
        except Exception:
            try:
                _med_nlp = medspacy.load(enable=[])
            except Exception:
                _med_nlp = medspacy.load()

        # Remove sectionizer if loaded — we use regex instead
        if "medspacy_sectionizer" in _med_nlp.pipe_names:
            _med_nlp.remove_pipe("medspacy_sectionizer")

        # Keep or add a sentence splitter — ConText requires sentence boundaries
        has_sent = any(n in _med_nlp.pipe_names
                       for n in ("medspacy_pyrush", "sentencizer", "senter"))
        if not has_sent:
            _med_nlp.add_pipe("sentencizer", first=True)

        # Ensure ConText is present
        if "medspacy_context" not in _med_nlp.pipe_names:
            _med_nlp.add_pipe("medspacy_context")

        # Remove NER if present (avoids entity conflicts with scispaCy)
        for name in ("ner", "medspacy_ner"):
            if name in _med_nlp.pipe_names:
                _med_nlp.remove_pipe(name)

        # Add case-report-specific negation cues not in the default ConText rules.
        # "Tuberculosis was excluded" → "excluded" appears AFTER the entity.
        try:
            from medspacy.context import ConTextRule
            context = _med_nlp.get_pipe("medspacy_context")
            context.add([
                ConTextRule("excluded",          "NEGATED_EXISTENCE", direction="BACKWARD"),
                ConTextRule("was excluded",      "NEGATED_EXISTENCE", direction="BACKWARD"),
                ConTextRule("were excluded",     "NEGATED_EXISTENCE", direction="BACKWARD"),
                ConTextRule("was not confirmed", "NEGATED_EXISTENCE", direction="BACKWARD"),
                ConTextRule("not confirmed",     "NEGATED_EXISTENCE", direction="BACKWARD"),
                ConTextRule("was not found",     "NEGATED_EXISTENCE", direction="BACKWARD"),
                ConTextRule("not present",       "NEGATED_EXISTENCE", direction="BACKWARD"),
                ConTextRule("not detected",      "NEGATED_EXISTENCE", direction="BACKWARD"),
            ])
        except Exception:
            pass

        _medspacy_ok = True

    except ImportError:
        print("[warn] medspaCy not installed — ConText negation filtering disabled")
        print("  pip install medspacy")
    except Exception as exc:
        print(f"[warn] medspaCy load failed: {exc}")


# ── Regex-based section detection ─────────────────────────────────────────────

def _regex_section_split(text: str) -> list[ClinicalSection]:
    """
    Section detection tuned for clinical case reports.
    Matches all-caps headings and known case-report keywords at line start.
    """
    heading_re = re.compile(
        r"(?m)^(?P<heading>"
        r"(?:[A-Z][A-Z ]{2,}|"   # all-caps line
        + "|".join(
            re.escape(lit)
            for _, lits in _SECTION_PATTERNS
            for lit in lits
        )
        + r"))\s*[:\n]",
        re.IGNORECASE,
    )

    matches = list(heading_re.finditer(text))
    if not matches:
        return [ClinicalSection("narrative", "narrative", text, 0, len(text))]

    sections = []
    for i, m in enumerate(matches):
        heading = m.group("heading").strip()
        body_start = m.end()
        body_end   = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body       = text[body_start:body_end].strip()

        category = "narrative"
        for cat, lits in _SECTION_PATTERNS:
            if any(heading.lower().startswith(lit.lower()) for lit in lits):
                category = cat
                break

        sections.append(ClinicalSection(
            category=category, title=heading,
            body=body, start_char=body_start, end_char=body_end,
        ))

    return sections


# ── ConText: inject scispaCy entities and get modifiers ──────────────────────

def run_context(
    text: str,
    ent_spans: list[tuple[int, int]],   # (start_char, end_char) from scispaCy
) -> dict[str, EntityModifiers]:
    """
    Run medspaCy ConText on the given entity spans from scispaCy.

    Returns a dict: span_text_lc → EntityModifiers.
    Empty dict if medspaCy is not available.
    """
    _load_medspacy()
    if not _medspacy_ok:
        return {}

    try:
        # Build doc and set sentence boundaries (ConText requires them).
        # Run the sentence splitter first, then inject entities, then ConText.
        doc = _med_nlp.make_doc(text[:100_000])
        for sent_pipe in ("medspacy_pyrush", "sentencizer", "senter"):
            if sent_pipe in _med_nlp.pipe_names:
                _med_nlp.get_pipe(sent_pipe)(doc)
                break

        new_ents = []
        for start_c, end_c in ent_spans:
            span = doc.char_span(start_c, end_c, label="ENTITY",
                                 alignment_mode="expand")
            if span is not None:
                new_ents.append(span)
        # Two scispaCy models can produce overlapping spans for the same text
        # region. filter_spans keeps the longest non-overlapping span.
        from spacy.util import filter_spans
        new_ents = filter_spans(new_ents)
        try:
            doc.set_ents(new_ents)
        except Exception:
            doc.ents = tuple(new_ents)

        ctx_pipe = _med_nlp.get_pipe("medspacy_context")
        ctx_pipe(doc)

        modifiers: dict[str, EntityModifiers] = {}
        for ent in doc.ents:
            key = ent.text.lower().strip()
            mods = EntityModifiers(
                is_negated      = getattr(ent._, "is_negated",      False),
                is_uncertain    = getattr(ent._, "is_uncertain",    False),
                is_historical   = getattr(ent._, "is_historical",   False),
                is_hypothetical = getattr(ent._, "is_hypothetical", False),
                is_family       = getattr(ent._, "is_family",       False),
            )
            modifiers[key] = mods

        return modifiers

    except Exception as exc:
        print(f"[warn] ConText run failed: {exc}")
        return {}


# ── Public API ────────────────────────────────────────────────────────────────

def preprocess_clinical_text(text: str) -> PreprocessedDoc:
    """
    Main entry point. Returns a PreprocessedDoc with:
      - sections: list of ClinicalSection (from regex detection)
      - entity_modifiers: empty at this stage (populated after scispaCy NER)
      - structured_for_llm: text with section headers suitable for Stage 1 prompt
    """
    sections = _regex_section_split(text)

    # Build structured text for Stage 1 LLM
    if len(sections) == 1 and sections[0].category == "narrative":
        structured = text
    else:
        parts = []
        for sec in sections:
            label = _SECTION_LABELS.get(sec.category, sec.category.upper())
            parts.append(f"[{label}]\n{sec.body}")
        structured = "\n\n".join(parts)

    return PreprocessedDoc(
        text=text,
        sections=sections,
        structured_for_llm=structured,
    )
