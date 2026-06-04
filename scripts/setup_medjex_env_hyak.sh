#!/bin/bash
# One-time setup for the MedJEx precompute environment on Hyak.
# Run interactively on a login node (no GPU needed):
#
#   bash scripts/setup_medjex_env_hyak.sh
#
# After this completes, convert the v1 model pack once (see step 2 below),
# then submit the precompute job.

set -euo pipefail

REPO_ROOT=/mmfs1/home/pgarg2/2026-LING-573-Summarization-Project
ENV_DIR=/gscratch/scrubbed/pgarg2/medjex-env
MEDCAT_DIR=/gscratch/scrubbed/pgarg2/medcat
MEDCAT_V2_DIR=/gscratch/scrubbed/pgarg2/medcat/umls_sm_v2

echo "=== MedJEx env setup ==="
echo "ENV_DIR:      $ENV_DIR"
echo "MEDCAT_DIR:   $MEDCAT_DIR"
echo "MEDCAT_V2_DIR:$MEDCAT_V2_DIR"

# ── Init submodule if not already populated ───────────────────────────────────
MEDJEX_DST="$REPO_ROOT/src/medjargone/MedJEx"
if [ ! -f "$MEDJEX_DST/README.md" ]; then
    echo "Initialising MedJEx submodule..."
    git -C "$REPO_ROOT" submodule update --init src/medjargone/MedJEx
else
    echo "MedJEx submodule already present: $MEDJEX_DST"
fi

# ── Conda env (stored in scrubbed to avoid home quota) ───────────────────────
eval "$(conda shell.bash hook)"

if [ ! -d "$ENV_DIR/conda-meta" ]; then
    echo "Creating conda env at $ENV_DIR..."
    conda create --prefix "$ENV_DIR" python=3.10 -y
else
    echo "Conda env already exists: $ENV_DIR"
fi
conda activate "$ENV_DIR"

# ── Install deps ──────────────────────────────────────────────────────────────
# PyTorch with CUDA 12.4 via the official conda channel
conda install pytorch pytorch-cuda=12.4 -c pytorch -c nvidia -y --quiet

python -m pip install -U pip setuptools wheel --quiet

# Pin spaCy/Thinc/MedSpaCy so they all support numpy>=2.0.
# medcat 2.8.1 requires numpy>=2.0; thinc>=8.3.4 was recompiled against numpy 2.x.
# medspacy 1.3.1 is the last release tested against spacy 3.8.2.
python -m pip install --no-cache-dir \
    "numpy>=2.0,<3" \
    "thinc>=8.3.4,<8.4" \
    "spacy==3.8.2" \
    "medcat[spacy]==2.8.1" \
    transformers nltk scikit-learn pandas tqdm torchcrf --quiet

# quickumls satisfies loader.py's unconditional top-level import; never called.
pip install quickumls --quiet

# ── Download NLTK data ────────────────────────────────────────────────────────
python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True)"

# ── Sanity check ─────────────────────────────────────────────────────────────
python - <<'PY'
import numpy, thinc, spacy, medcat
from medcat.cat import CAT
print(f"numpy   {numpy.__version__}")
print(f"thinc   {thinc.__version__}")
print(f"spacy   {spacy.__version__}")
print(f"medcat  {medcat.__version__}")
print("CAT import OK")
PY

# ── Convert v1 model pack to v2 (one-time, skip if already done) ─────────────
V1_ZIP="$MEDCAT_DIR/umls_sm_pt2ch_533bab5115c6c2d6.zip"
if [ -f "$V1_ZIP" ] && [ ! -d "$MEDCAT_V2_DIR" ]; then
    echo "Converting v1 model pack to v2 format (one-time, ~5 min)..."
    mkdir -p "$MEDCAT_V2_DIR"
    python -m medcat.utils.legacy.legacy_converter \
        "$V1_ZIP" \
        "$MEDCAT_V2_DIR" \
        --verbose
    echo "Converted model at: $MEDCAT_V2_DIR"
elif [ -d "$MEDCAT_V2_DIR" ]; then
    echo "v2 model pack already present: $MEDCAT_V2_DIR"
else
    echo "[warn] v1 zip not found at $V1_ZIP — skipping conversion."
    echo "       Download the model pack first, then re-run this script."
fi

mkdir -p "$MEDCAT_DIR"

echo ""
echo "=== Setup complete ==="
echo ""
echo "Next steps:"
echo "  1. If conversion was skipped above, download the v1 model pack to:"
echo "       $V1_ZIP"
echo "     then re-run this script to trigger conversion."
echo ""
echo "  2. Submit the precompute job:"
echo "       sbatch scripts/run_precompute_medjex_hyak.sh \\"
echo "           --medcat-path $MEDCAT_V2_DIR"
