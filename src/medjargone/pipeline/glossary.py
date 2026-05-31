"""
Stage 2 — UMLS-grounded glossary subsystem.

Input : source text (str)
Output: list[GlossaryEntry]

Resolution order for each candidate term:
  1. FILTER  — drop if semantic type is non-medical (via UMLS)
  2. ACRONYM — expand only if AbbreviationDetector found a source definition
  3. NEEDS-GLOSS? — skip if CHV consumer term == source term (already lay)
  4. GLOSS (priority order):
       (i)   CHV consumer-preferred atom  (UMLS atoms, sab=CHV, ispref=Y)
       (ii)  UMLS definition (MSH/NCI/MedlinePlus), optionally truncated
       (iii) ClinicalTables/MedlinePlus Connect fallback (optional, online)
       (iv)  none → keep term + semantic type description; confidence=low
  5. TAG  — attach UMLS semantic type (never let rewriter change drug→technique)
  6. DRUG — normalize via RxNorm to attach RxCUI for Stage-4 identity check
  7. ANATOMY — resolve via SNOMED CT (if local subset built) + curated lay map

Offline by default when umls_subset.sqlite is present. UTS API is the fallback
(and the primary path during development before the subset is built).
"""

from __future__ import annotations

import json
import re
import sqlite3
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import requests

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from medjargone import config

# ── NLP models (lazy) ─────────────────────────────────────────────────────────
_nlp_bc5cdr  = None
_nlp_bionlp  = None
_abbrev_ok   = False


# ── Data types ────────────────────────────────────────────────────────────────

@dataclass
class GlossaryEntry:
    term: str
    cui: Optional[str]
    semantic_type: Optional[str]
    rxcui: Optional[str]
    snomed: Optional[str]
    gloss: Optional[str]
    gloss_source: Optional[str]
    needs_gloss: bool
    confidence: str                  # "high" | "medium" | "low"
    source_context: str
    section: Optional[str] = None    # clinical section the term came from
    instruction: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


# ── API cache (SQLite) ─────────────────────────────────────────────────────────

class APICache:
    def __init__(self, db_path: Path = config.UTS_CACHE_DB):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False, timeout=30)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, value TEXT)
        """)
        self._conn.commit()

    def get(self, key: str) -> Optional[str]:
        row = self._conn.execute(
            "SELECT value FROM cache WHERE key=?", (key,)
        ).fetchone()
        return row[0] if row else None

    def set(self, key: str, value: str) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO cache VALUES (?,?)", (key, value)
        )
        self._conn.commit()


# ── UMLS local subset index ───────────────────────────────────────────────────

class UMLSIndex:
    """
    Read-only wrapper around umls_subset.sqlite.
    Present when M6 (build_umls_index) has been run; absent during early dev.
    """

    def __init__(self, db_path: Path = config.UMLS_SUBSET_DB):
        if not db_path.exists():
            raise FileNotFoundError(db_path)
        self._conn = sqlite3.connect(
            f"file:{db_path}?mode=ro", uri=True, check_same_thread=False
        )
        self._conn.row_factory = sqlite3.Row

    def find_cui(self, term: str) -> Optional[str]:
        """Return the best CUI for a term string (English, non-suppressed)."""
        term_lc = term.lower().strip()
        # Prefer preferred atoms (ispref=Y) then fall through
        for ispref in ("Y", "N"):
            row = self._conn.execute(
                "SELECT cui FROM atoms WHERE str_lc=? AND ispref=? LIMIT 1",
                (term_lc, ispref),
            ).fetchone()
            if row:
                return row["cui"]
        return None

    def semantic_types(self, cui: str) -> list[str]:
        rows = self._conn.execute(
            "SELECT sty FROM semtypes WHERE cui=?", (cui,)
        ).fetchall()
        return [r["sty"] for r in rows]

    def chv_gloss(self, cui: str) -> Optional[str]:
        """Return the CHV consumer-preferred atom string for a CUI."""
        row = self._conn.execute(
            "SELECT str FROM atoms WHERE cui=? AND sab=? AND ispref='Y' LIMIT 1",
            (cui, config.CHV_SAB),
        ).fetchone()
        return row["str"] if row else None

    def definition(self, cui: str) -> Optional[tuple[str, str]]:
        """Return (def_text, sab) from the highest-priority definition source."""
        for sab in config.DEF_PRIORITY_SABS:
            row = self._conn.execute(
                "SELECT def_text FROM definitions WHERE cui=? AND sab=? LIMIT 1",
                (cui, sab),
            ).fetchone()
            if row:
                return row["def_text"], sab
        return None

    def snomed_cui(self, term: str) -> Optional[str]:
        """Return SNOMEDCT_US CUI for an anatomy/body-structure term."""
        term_lc = term.lower().strip()
        row = self._conn.execute(
            "SELECT cui FROM atoms WHERE str_lc=? AND sab='SNOMEDCT_US' LIMIT 1",
            (term_lc,),
        ).fetchone()
        return row["cui"] if row else None


# ── UTS REST API client ───────────────────────────────────────────────────────

class UTSClient:
    """
    Minimal UTS REST client. Auth = ?apiKey=... on every request.
    Caches every successful response to api_cache.sqlite.
    """

    def __init__(self, api_key: str = config.UMLS_API_KEY,
                 cache: Optional[APICache] = None):
        self._key = api_key
        self._cache = cache or APICache()

    def _get(self, url: str, params: dict) -> Optional[dict]:
        import urllib.parse
        params = {**params, "apiKey": self._key}
        key = url + "?" + urllib.parse.urlencode(sorted(params.items()))
        cached = self._cache.get(key)
        if cached is not None:
            return json.loads(cached)

        for attempt in range(config.API_MAX_RETRIES + 1):
            try:
                resp = requests.get(url, params=params, timeout=config.API_TIMEOUT)
                resp.raise_for_status()
                data = resp.json()
                self._cache.set(key, json.dumps(data))
                return data
            except Exception:
                if attempt < config.API_MAX_RETRIES:
                    time.sleep(config.API_RETRY_DELAY)
        return None

    # Vocabularies that contain canonical disease/drug/anatomy CUIs.
    # Restricting search to these prevents UMLS returning non-medical CUIs
    # (e.g. LOINC observation codes) as the top hit for disease terms.
    _MEDICAL_SABS = "SNOMEDCT_US,MSH,NCI,RXNORM,CHV,MED-RT"

    def search(self, term: str) -> Optional[str]:
        """Return best CUI for term via /search/current.

        Prefers authoritative medical vocabularies (SNOMED, MeSH, NCI, RxNorm)
        so that disease terms resolve to Disease-or-Syndrome CUIs rather than
        LOINC/observation codes that share the same surface form.
        """
        for search_type in ("exact", "words"):
            data = self._get(config.UTS_SEARCH_URL, {
                "string": term, "searchType": search_type,
                "returnIdType": "concept", "pageSize": 5,
                "sabs": self._MEDICAL_SABS,
            })
            results = (data or {}).get("result", {}).get("results", [])
            cui = next((r["ui"] for r in results if r.get("ui") != "NONE"), None)
            if cui:
                return cui
        # Last resort: unrestricted search
        data = self._get(config.UTS_SEARCH_URL, {
            "string": term, "searchType": "words",
            "returnIdType": "concept", "pageSize": 1,
        })
        results = (data or {}).get("result", {}).get("results", [])
        return results[0]["ui"] if results and results[0].get("ui") != "NONE" else None

    def semantic_types(self, cui: str) -> list[str]:
        data = self._get(
            config.UTS_CONCEPT_URL.format(cui=cui), {}
        )
        stys = (data or {}).get("result", {}).get("semanticTypes", [])
        return [s.get("name", "") for s in stys]

    def chv_gloss(self, cui: str) -> Optional[str]:
        data = self._get(
            config.UTS_ATOMS_URL.format(cui=cui),
            {"sabs": config.CHV_SAB, "pageSize": 10},
        )
        atoms = (data or {}).get("result", [])
        # Prefer preferred atoms
        for atom in sorted(atoms, key=lambda a: a.get("termType") != "PT"):
            name = atom.get("name", "").strip()
            if name:
                return name
        return None

    def definition(self, cui: str) -> Optional[tuple[str, str]]:
        data = self._get(config.UTS_DEFS_URL.format(cui=cui), {"pageSize": 20})
        defs = (data or {}).get("result", [])
        priority = {sab: i for i, sab in enumerate(config.DEF_PRIORITY_SABS)}
        defs_sorted = sorted(
            defs, key=lambda d: priority.get(d.get("rootSource", ""), 999)
        )
        for d in defs_sorted:
            text = d.get("value", "").strip()
            if text:
                return text, d.get("rootSource", "UMLS")
        return None


# ── RxNorm lookup ──────────────────────────────────────────────────────────────

def rxnorm_lookup(term: str, cache: APICache) -> Optional[str]:
    """Return RxCUI for a drug name, or None. Uses free RxNorm API (no key)."""
    import urllib.parse
    key = f"rxnorm:{term.lower()}"
    cached = cache.get(key)
    if cached is not None:
        return cached if cached != "__none__" else None

    try:
        resp = requests.get(
            config.RXNORM_FIND_URL,
            params={"name": term, "search": "1"},
            timeout=config.API_TIMEOUT,
        )
        resp.raise_for_status()
        rxcui = resp.json().get("idGroup", {}).get("rxnormId", [None])[0]
        cache.set(key, rxcui if rxcui else "__none__")
        return rxcui
    except Exception:
        return None


# ── Candidate filtering ────────────────────────────────────────────────────────

_BLOCKLIST = re.compile(
    r"^(our|the|this|a|an|patient|case|report|figure|table|"
    r"university|hospital|department|et\sal|ibid|"
    r"male|female|man|woman|boy|girl|child|adult|"
    r"day|week|month|year|age|old)\b",
    re.IGNORECASE,
)


def _is_medical(sty_list: list[str]) -> bool:
    return bool(set(sty_list) & config.MEDICAL_SEMANTIC_TYPES)


# ── scispaCy loading ───────────────────────────────────────────────────────────

def _load_nlp():
    global _nlp_bc5cdr, _nlp_bionlp, _abbrev_ok
    if _nlp_bc5cdr is not None:
        return
    try:
        import spacy
        from scispacy.abbreviation import AbbreviationDetector
        _nlp_bc5cdr = spacy.load(config.SCISPACY_BC5CDR_MODEL,
                                  disable=["parser", "senter"])
        _nlp_bionlp = spacy.load(config.SCISPACY_BIONLP_MODEL,
                                  disable=["parser", "senter"])
        _nlp_bc5cdr.add_pipe("abbreviation_detector")
        _nlp_bionlp.add_pipe("abbreviation_detector")
        _abbrev_ok = True
    except ImportError:
        print("[warn] scispaCy not installed — NER disabled")
    except OSError as exc:
        print(f"[warn] scispaCy model load failed: {exc}")


def _extract_candidates(
    text: str,
) -> tuple[list[tuple[str, str, str, int, int]], dict[str, str]]:
    """
    Returns:
      candidates: [(span_text, ner_type, context_snippet, start_char, end_char), …]
      abbrev_map: {ACRONYM_UPPER: long_form}  — source-defined only
    """
    _load_nlp()
    candidates: list[tuple[str, str, str, int, int]] = []
    abbrev_map: dict[str, str] = {}

    if _nlp_bc5cdr is None:
        return candidates, abbrev_map

    seen: set[str] = set()
    for nlp in (_nlp_bc5cdr, _nlp_bionlp):
        doc = nlp(text[:100_000])

        if not abbrev_map and _abbrev_ok:
            for abrv in doc._.abbreviations:
                short = str(abrv).upper()
                long  = str(abrv._.long_form).strip()
                if long:
                    abbrev_map[short] = long

        for ent in doc.ents:
            span = ent.text.strip()
            if not span or len(span) < 2 or _BLOCKLIST.match(span):
                continue
            key = span.lower()
            if key in seen:
                continue
            seen.add(key)
            ctx_start = max(0, ent.start_char - 40)
            ctx_end   = min(len(text), ent.end_char + 40)
            context   = "…" + text[ctx_start:ctx_end].strip() + "…"
            candidates.append((span, ent.label_, context, ent.start_char, ent.end_char))

    return candidates, abbrev_map


# ── Per-term glossing ─────────────────────────────────────────────────────────

def _gloss_one(
    term: str,
    ner_type: str,
    context: str,
    umls_index: Optional[UMLSIndex],
    uts_client: Optional[UTSClient],
    cache: APICache,
    abbrev_map: dict[str, str],
    curated: dict[str, str],
) -> GlossaryEntry:

    # ── Acronym check ─────────────────────────────────────────────────────────
    is_acronym = bool(re.match(r"^[A-Z]{2,6}$", term))
    if is_acronym:
        if term in abbrev_map:
            term = abbrev_map[term]   # expand; continue glossing the long form
        else:
            return GlossaryEntry(
                term=term, cui=None, semantic_type=None,
                rxcui=None, snomed=None, gloss=None, gloss_source=None,
                needs_gloss=True, confidence="low", source_context=context,
                instruction=(
                    f"Do NOT expand or define '{term}'; render as "
                    f"'{term}, a term used in the report'."
                ),
            )

    # ── CUI resolution ────────────────────────────────────────────────────────
    cui: Optional[str] = None
    if umls_index is not None:
        cui = umls_index.find_cui(term)
    if cui is None and uts_client is not None:
        cui = uts_client.search(term)

    # ── Semantic type + medical filter ────────────────────────────────────────
    stys: list[str] = []
    if cui:
        if umls_index is not None:
            stys = umls_index.semantic_types(cui)
        if not stys and uts_client is not None:
            stys = uts_client.semantic_types(cui)

    if cui and not _is_medical(stys):
        # Non-medical concept — skip entirely
        return GlossaryEntry(
            term=term, cui=cui, semantic_type=stys[0] if stys else None,
            rxcui=None, snomed=None, gloss=None, gloss_source=None,
            needs_gloss=False, confidence="high", source_context=context,
        )

    sty_str = stys[0] if stys else ner_type

    # ── Curated anatomy map ───────────────────────────────────────────────────
    term_lc = term.lower()
    if term_lc in curated:
        snomed = None
        if umls_index is not None:
            snomed = umls_index.snomed_cui(term)
        return GlossaryEntry(
            term=term, cui=cui, semantic_type=sty_str,
            rxcui=None, snomed=snomed, gloss=curated[term_lc],
            gloss_source="curated", needs_gloss=True, confidence="high",
            source_context=context,
        )

    # ── RxNorm for drugs ──────────────────────────────────────────────────────
    drug_stys = {
        "Pharmacologic Substance", "Clinical Drug", "Antibiotic",
        "Organic Chemical", "Biologically Active Substance",
    }
    is_drug = bool(set(stys) & drug_stys) or ner_type in ("CHEMICAL", "SIMPLE_CHEMICAL")
    rxcui: Optional[str] = None
    if is_drug:
        rxcui = rxnorm_lookup(term, cache)

    # ── SNOMED for anatomy ────────────────────────────────────────────────────
    anatomy_stys = {
        "Body Part, Organ, or Organ Component", "Anatomical Structure",
        "Body Space or Junction", "Body System", "Tissue", "Cell",
    }
    is_anatomy = bool(set(stys) & anatomy_stys) or ner_type in ("ORGAN", "TISSUE")
    snomed_id: Optional[str] = None
    if is_anatomy and umls_index is not None:
        snomed_id = umls_index.snomed_cui(term)

    # ── NEEDS-GLOSS check ─────────────────────────────────────────────────────
    # Fetch CHV gloss first; if it's identical to the source term → already lay
    chv_gloss: Optional[str] = None
    if cui:
        if umls_index is not None:
            chv_gloss = umls_index.chv_gloss(cui)
        if chv_gloss is None and uts_client is not None:
            chv_gloss = uts_client.chv_gloss(cui)

    if chv_gloss and chv_gloss.lower() == term_lc:
        return GlossaryEntry(
            term=term, cui=cui, semantic_type=sty_str,
            rxcui=rxcui, snomed=snomed_id, gloss=None, gloss_source=None,
            needs_gloss=False, confidence="high", source_context=context,
        )

    # ── GLOSS priority (i): CHV consumer term ─────────────────────────────────
    if chv_gloss:
        return GlossaryEntry(
            term=term, cui=cui, semantic_type=sty_str,
            rxcui=rxcui, snomed=snomed_id, gloss=chv_gloss,
            gloss_source="CHV", needs_gloss=True, confidence="high",
            source_context=context,
        )

    # ── GLOSS priority (ii): UMLS definition ─────────────────────────────────
    def_result: Optional[tuple[str, str]] = None
    if cui:
        if umls_index is not None:
            def_result = umls_index.definition(cui)
        if def_result is None and uts_client is not None:
            def_result = uts_client.definition(cui)

    if def_result:
        def_text, def_sab = def_result
        if len(def_text) > 150:
            cut = def_text.rfind(" ", 0, 150)
            def_text = def_text[: cut if cut > 0 else 150] + "…"
        return GlossaryEntry(
            term=term, cui=cui, semantic_type=sty_str,
            rxcui=rxcui, snomed=snomed_id, gloss=def_text,
            gloss_source=def_sab, needs_gloss=True, confidence="medium",
            source_context=context,
        )

    # ── GLOSS priority (iii): ClinicalTables/MedlinePlus fallback ─────────────
    if not is_drug:
        ct_gloss = _clinical_tables_gloss(term, cache)
        if ct_gloss:
            return GlossaryEntry(
                term=term, cui=cui, semantic_type=sty_str,
                rxcui=rxcui, snomed=snomed_id, gloss=ct_gloss,
                gloss_source="ClinicalTables", needs_gloss=True, confidence="medium",
                source_context=context,
            )

    # ── GLOSS priority (iv): no gloss ─────────────────────────────────────────
    return GlossaryEntry(
        term=term, cui=cui, semantic_type=sty_str,
        rxcui=rxcui, snomed=snomed_id, gloss=None, gloss_source=None,
        needs_gloss=True, confidence="low", source_context=context,
    )


# ── Optional ClinicalTables fallback ─────────────────────────────────────────

def _clinical_tables_gloss(term: str, cache: APICache) -> Optional[str]:
    key = f"ct:{term.lower()}"
    cached = cache.get(key)
    if cached is not None:
        return cached if cached != "__none__" else None
    try:
        resp = requests.get(
            f"{config.CLINICAL_TABLES_BASE}/conditions/v3/search",
            params={"terms": term, "maxList": 1, "df": "consumer_name"},
            timeout=config.API_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        name = data[2][0][0] if data[0] else None
        cache.set(key, name if name else "__none__")
        return name
    except Exception:
        return None


# ── Public API ─────────────────────────────────────────────────────────────────

def build_glossary(
    source_text: str,
    umls_index: Optional[UMLSIndex] = None,
    uts_client: Optional[UTSClient] = None,
    cache: Optional[APICache] = None,
    curated_path: Path = config.CURATED_ANATOMY,
) -> list[GlossaryEntry]:
    """
    Stage 2 entry point. Returns GlossaryEntry list for every medical term in text.

    umls_index / uts_client : shared across batch runs (init once in run.py).
    """
    if cache is None:
        cache = APICache()

    if umls_index is None and config.UMLS_SUBSET_DB.exists():
        umls_index = UMLSIndex()

    if uts_client is None and config.UMLS_API_KEY:
        uts_client = UTSClient(cache=cache)

    curated: dict[str, str] = {}
    if curated_path.exists():
        with open(curated_path, encoding="utf-8") as fh:
            curated = json.load(fh)

    candidates, abbrev_map = _extract_candidates(source_text)

    entries: list[GlossaryEntry] = []
    seen: set[str] = set()

    for term, ner_type, context, start_c, end_c in candidates:
        key = term.lower()
        if key in seen:
            continue
        seen.add(key)

        entry = _gloss_one(
            term, ner_type, context,
            umls_index, uts_client, cache,
            abbrev_map, curated,
        )
        entries.append(entry)

    return entries
