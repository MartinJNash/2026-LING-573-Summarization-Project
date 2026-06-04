#!/bin/bash
# One-time setup for the MedJEx precompute environment on Hyak.
# Run interactively on a login node (no GPU needed):
#
#   bash scripts/setup_medjex_env_hyak.sh
#
# After this completes, submit the precompute job:
#   sbatch scripts/run_precompute_medjex_hyak.sh

set -euo pipefail

REPO_ROOT=/mmfs1/home/pgarg2/2026-LING-573-Summarization-Project
ENV_DIR=/gscratch/scrubbed/pgarg2/medjex-env
MEDCAT_DIR=/gscratch/scrubbed/pgarg2/medcat

echo "=== MedJEx env setup ==="
echo "ENV_DIR:     $ENV_DIR"
echo "MEDCAT_DIR:  $MEDCAT_DIR"

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

pip install transformers medspacy nltk scikit-learn pandas tqdm medcat --quiet
# quickumls satisfies loader.py's top-level import; we never call it.
pip install quickumls --quiet
# thinc 8.x was compiled against numpy <2.0; pin to avoid binary incompatibility
pip install "numpy<2.0.0" --quiet

# ── Download NLTK data ────────────────────────────────────────────────────────
python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True)"

# ── Download spaCy model used by medspacy ─────────────────────────────────────
python -m spacy download en_core_web_sm --quiet

# ── Create MedCAT model dir ───────────────────────────────────────────────────
mkdir -p "$MEDCAT_DIR"

echo ""
echo "=== Setup complete ==="
echo ""
echo "Next steps:"
echo "  1. Download a MedCAT UMLS model pack into $MEDCAT_DIR/"
echo "     e.g.  mc_modelpack_snomed_int_16_mar_2022_25be3857ba34bdd5.zip"
echo "     from  https://huggingface.co/medcat (requires HuggingFace login)"
echo "     Unzip so the .zip or unpacked dir is at:"
echo "       $MEDCAT_DIR/<model_pack_name>"
echo ""
echo "  2. Submit the precompute job:"
echo "       sbatch scripts/run_precompute_medjex_hyak.sh \\"
echo "           --medcat-path $MEDCAT_DIR/<model_pack_name>"
