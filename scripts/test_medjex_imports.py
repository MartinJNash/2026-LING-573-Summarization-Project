"""
Quick smoke-test for all MedJEx imports and dependencies.
Run from the repo root with the medjex-env activated:

    conda activate /gscratch/scrubbed/pgarg2/medjex-env
    python scripts/test_medjex_imports.py

Delete this file when done.
"""

import sys
from pathlib import Path

REPO_ROOT  = Path(__file__).parents[1]
MEDJEX_DIR = REPO_ROOT / "src" / "medjargone" / "MedJEx"

if str(MEDJEX_DIR) not in sys.path:
    sys.path.insert(0, str(MEDJEX_DIR))

OK  = "OK "
ERR = "ERR"

checks = [
    ("numpy >= 2.0",
     "import numpy; assert tuple(int(x) for x in numpy.__version__.split('.')[:2]) >= (2,0), numpy.__version__; print(f'     numpy {numpy.__version__}')"),

    ("torch",
     "import torch; print(f'     torch {torch.__version__}  CUDA={torch.cuda.is_available()}')"),

    ("transformers AutoTokenizer",
     "from transformers import AutoTokenizer, AutoConfig; print('     transformers OK')"),

    ("torchcrf",
     "from torchcrf import CRF; print('     torchcrf OK')"),

    ("nltk sent_tokenize + word_tokenize",
     "from nltk.tokenize import sent_tokenize, word_tokenize; sent_tokenize('Hello world.')"),

    ("medcat CAT import",
     "from medcat.cat import CAT; print('     medcat OK')"),

    ("quickumls (stub import only)",
     "from quickumls import QuickUMLS; print('     quickumls OK')"),

    ("loader.loader",
     "from loader.loader import load_file, load_data, load_ner_file"),

    ("utils.sequence_labeler",
     "from utils.sequence_labeler import SequenceLabeler"),

    ("utils.term_weighting",
     "from utils.term_weighting import TermFrequency, MLM_weight"),

    ("utils.normalization",
     "from utils.normalization import Normalizer"),

    ("models.models EarlyAndLateFusionBertMLPCRF",
     "from models.models import EarlyAndLateFusionBertMLPCRF, EarlyAndLateFusionrobertaMLPCRF"),

    ("MedCAT_wrapper class",
     "from utils.MedCAT import MedCAT_wrapper; print('     MedCAT_wrapper OK')"),

    ("precompute_medjex_spans top-level imports",
     """
import importlib.util, pathlib
spec = importlib.util.spec_from_file_location(
    'precompute', str(pathlib.Path('scripts/precompute_medjex_spans.py'))
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
print('     precompute_medjex_spans.py loaded OK')
"""),
]

print("=" * 55)
print("MedJEx import smoke-test")
print("=" * 55)

failures = []
for label, code in checks:
    try:
        exec(compile(code.strip(), "<check>", "exec"))
        print(f"{OK}  {label}")
    except Exception as exc:
        print(f"{ERR}  {label}")
        print(f"     {type(exc).__name__}: {exc}")
        failures.append(label)

print("=" * 55)
if failures:
    print(f"FAILED ({len(failures)}/{len(checks)}): {', '.join(failures)}")
    sys.exit(1)
else:
    print(f"All {len(checks)} checks passed — safe to sbatch.")
