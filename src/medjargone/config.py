"""
Central configuration for the MedJarGone v4 pipeline (UMLS edition).
All paths are relative to the project root (two levels above this file).
UMLS_API_KEY must be set as an environment variable — never hard-coded.
"""

import os
from pathlib import Path

# Project root: src/medjargone/config.py → ../../
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# ── Data directories ──────────────────────────────────────────────────────────
DATA_DIR = PROJECT_ROOT / "data" / "medjargone"
CACHE_DIR = DATA_DIR / "cache"

UMLS_SUBSET_DB   = DATA_DIR / "umls_subset.sqlite"
UTS_CACHE_DB     = CACHE_DIR / "uts_cache.sqlite"
CURATED_ANATOMY  = Path(__file__).resolve().parent / "resources" / "curated_anatomy.json"

# ── MultiClinSum data ─────────────────────────────────────────────────────────
MULTICLINSUM_TEST_DIR  = PROJECT_ROOT / "data" / "multiclinsum_test_en"
MULTICLINSUM_TRAIN_DIR = PROJECT_ROOT / "data" / "multiclinsum_gs_train_en"

# ── UTS / UMLS API ────────────────────────────────────────────────────────────
UMLS_API_KEY  = os.environ.get("UMLS_API_KEY", "")   # set in shell: export UMLS_API_KEY=...
UTS_BASE_URL = "https://uts-ws.nlm.nih.gov/rest"

# Convenience endpoint templates (format with .format(cui=..., apiKey=UMLS_API_KEY))
UTS_SEARCH_URL    = UTS_BASE_URL + "/search/current"
UTS_CONCEPT_URL   = UTS_BASE_URL + "/content/current/CUI/{cui}"
UTS_ATOMS_URL     = UTS_BASE_URL + "/content/current/CUI/{cui}/atoms"
UTS_DEFS_URL      = UTS_BASE_URL + "/content/current/CUI/{cui}/definitions"

# SAB filter for CHV atoms (lay glosses)
CHV_SAB = "CHV"

# Preferred definition sources, in priority order
DEF_PRIORITY_SABS = ["MSH", "NCI", "MedlinePlus", "CHV"]

# ── RxNorm (same UTS key) ─────────────────────────────────────────────────────
RXNORM_BASE_URL = "https://rxnav.nlm.nih.gov/REST"
RXNORM_FIND_URL = RXNORM_BASE_URL + "/rxcui.json"

# ── scispaCy models ───────────────────────────────────────────────────────────
SCISPACY_BC5CDR_MODEL = "en_ner_bc5cdr_md"
SCISPACY_BIONLP_MODEL = "en_ner_bionlp13cg_md"

# ── medspaCy ──────────────────────────────────────────────────────────────────
# medspaCy uses en_core_web_sm as its base model; install with:
#   pip install medspacy
#   python -m spacy download en_core_web_sm
MEDSPACY_BASE_MODEL = "en_core_web_sm"

# Entities tagged with these ConText modifiers are excluded from the glossary
# (we never gloss "pneumonia" in "no evidence of pneumonia")
CONTEXT_EXCLUDE_MODIFIERS = {"NEGATED_EXISTENCE", "POSSIBLE_EXISTENCE",
                              "HISTORICAL", "HYPOTHETICAL", "FAMILY"}

# ── LLM (Ollama) ─────────────────────────────────────────────────────────────
LLM_MODEL          = "qwen2.5:3b"
LLM_MAX_NEW_TOKENS = 512
OLLAMA_HOST        = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

# ── Stage 4 — MiniCheck ──────────────────────────────────────────────────────
# Use Flan-T5-Large locally (CPU-friendly); swap to Bespoke-7B if GPU available.
MINICHECK_MODEL = "lytang/MiniCheck-Flan-T5-Large"

MINICHECK_TAU_LOW  = 0.35   # below this  → confident fail
MINICHECK_TAU_HIGH = 0.65   # above this  → confident pass (routes to Tier-3 below)

# ── Stage 4 — Tier 3 clinical NLI arbiter ────────────────────────────────────
MEDNLI_MODEL = "pritamdeka/PubMedBERT-MNLI-MedNLI"
MEDNLI_ENTAILMENT_THRESHOLD = 0.50

# ── Optional online fallback (ClinicalTables / MedlinePlus Connect) ───────────
CLINICAL_TABLES_BASE     = "https://clinicaltables.nlm.nih.gov/api"
MEDLINEPLUS_CONNECT_URL  = "https://connect.medlineplus.gov/service"
OID_ICD10CM              = "2.16.840.1.113883.6.90"
OID_RXCUI                = "2.16.840.1.113883.6.88"

# ── API request settings ──────────────────────────────────────────────────────
API_TIMEOUT     = 8      # seconds per request
API_MAX_RETRIES = 2
API_RETRY_DELAY = 1.0

# ── UMLS semantic types to keep (kills demographics/geography/pure chemistry) ─
MEDICAL_SEMANTIC_TYPES = {
    "Disease or Syndrome",
    "Neoplastic Process",
    "Sign or Symptom",
    "Finding",
    "Mental or Behavioral Dysfunction",
    "Injury or Poisoning",
    "Pharmacologic Substance",
    "Clinical Drug",
    "Antibiotic",
    "Organic Chemical",
    "Immunologic Factor",
    "Biologically Active Substance",
    "Hormone",
    "Enzyme",
    "Vitamin",
    "Body Part, Organ, or Organ Component",
    "Anatomical Structure",
    "Body Space or Junction",
    "Body System",
    "Tissue",
    "Cell",
    "Body Substance",
    "Diagnostic Procedure",
    "Laboratory Procedure",
    "Therapeutic or Preventive Procedure",
    "Medical Device",
    "Laboratory or Test Result",
    "Pathologic Function",
    "Physiologic Function",
    "Organism Function",
}
