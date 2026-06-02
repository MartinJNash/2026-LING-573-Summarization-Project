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
HF_CACHE=/gscratch/scrubbed/pgarg2/hf-cache

echo "=== MedJEx env setup ==="
echo "ENV_DIR:     $ENV_DIR"
echo "MEDCAT_DIR:  $MEDCAT_DIR"

# ── Clone MedJEx repo if not present ─────────────────────────────────────────
MEDJEX_DST="$REPO_ROOT/src/medjargone/MedJEx"
if [ ! -d "$MEDJEX_DST" ]; then
    echo "Cloning MedJEx repo…"
    git clone https://github.com/EMNLP2022-MedJEx/MedJEx.git "$MEDJEX_DST"
else
    echo "MedJEx repo already present: $MEDJEX_DST"
fi

# ── Create Python venv ────────────────────────────────────────────────────────
if [ ! -d "$ENV_DIR" ]; then
    echo "Creating venv at $ENV_DIR…"
    python3 -m venv "$ENV_DIR"
fi
source "$ENV_DIR/bin/activate"

pip install --upgrade pip --quiet

# ── Install deps ──────────────────────────────────────────────────────────────
# torch with CUDA 12.4 (matches Hyak klone L40/A100 nodes)
pip install torch --index-url https://download.pytorch.org/whl/cu124 --quiet
pip install transformers medspacy nltk scikit-learn pandas tqdm medcat --quiet
# quickumls must be installed to satisfy loader.py's top-level import;
# we never actually use it (MedCAT is the matcher).
pip install quickumls --quiet

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
