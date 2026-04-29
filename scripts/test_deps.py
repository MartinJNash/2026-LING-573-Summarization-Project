"""
Quick dependency check — run before submitting batch jobs.
Tests all imports and basic functionality used in inference + eval.
"""

import sys
import traceback

PASS = []
FAIL = []

def check(name, fn):
    try:
        result = fn()
        msg = f"  {result}" if result else ""
        print(f"[OK]  {name}{msg}")
        PASS.append(name)
    except Exception as e:
        print(f"[FAIL] {name}: {e}")
        FAIL.append((name, str(e)))


# --- Core ---
check("python version",     lambda: sys.version)
check("torch",              lambda: __import__("torch").__version__)
check("torch CUDA",         lambda: f"cuda={'available' if __import__('torch').cuda.is_available() else 'NOT available'}")
check("numpy",              lambda: __import__("numpy").__version__)
check("transformers",       lambda: __import__("transformers").__version__)
check("huggingface_hub",    lambda: __import__("huggingface_hub").__version__)

# --- Inference ---
check("peft",               lambda: __import__("peft").__version__)
check("peft PeftModel",     lambda: __import__("peft", fromlist=["PeftModel"]).PeftModel and "ok")
check("peft PeftConfig",    lambda: __import__("peft", fromlist=["PeftConfig"]).PeftConfig and "ok")
check("safetensors",        lambda: __import__("safetensors").__version__)
check("accelerate",         lambda: __import__("accelerate").__version__)

# --- Data ---
check("datasets",           lambda: __import__("datasets").__version__)
check("pandas",             lambda: __import__("pandas").__version__)

# --- Eval ---
check("evaluate",           lambda: __import__("evaluate").__version__)
check("bert_score",         lambda: __import__("bert_score").__version__ if hasattr(__import__("bert_score"), "__version__") else "ok")
check("rouge_score",        lambda: __import__("rouge_score") and "ok")
check("nltk",               lambda: __import__("nltk").__version__)
check("textstat",           lambda: __import__("textstat").__version__ if hasattr(__import__("textstat"), "__version__") else "ok")

# --- SummaC ---
check("summac import",      lambda: __import__("summac") and "ok")
check("summac SummaCZS",    lambda: __import__("summac.model_summac", fromlist=["SummaCZS"]).SummaCZS and "ok")

# --- spacy / scispacy ---
check("spacy",              lambda: __import__("spacy").__version__)
check("scispacy",           lambda: __import__("scispacy").__version__)
check("thinc",              lambda: __import__("thinc").__version__)
check("spacy en_core_sci_sm", lambda: __import__("spacy").load("en_core_sci_sm") and "ok")

# --- Project modules ---
check("read_data",          lambda: __import__("read_data") and "ok")
check("model (Summarizer)", lambda: __import__("model", fromlist=["Summarizer"]).Summarizer and "ok")

# --- Summary ---
print()
print(f"{'='*40}")
print(f"  {len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    print("\nFailed checks:")
    for name, err in FAIL:
        print(f"  - {name}: {err}")
    sys.exit(1)
else:
    print("  All good — safe to submit jobs.")
print(f"{'='*40}")
