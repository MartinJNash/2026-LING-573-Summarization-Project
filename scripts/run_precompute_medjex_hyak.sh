#!/bin/bash
# Precompute MedJEx jargon spans for the full MultiClinSum test split on Hyak.
#
# First run setup_medjex_env_hyak.sh once, then:
#   sbatch scripts/run_precompute_medjex_hyak.sh \
#       --medcat-path /gscratch/scrubbed/pgarg2/medcat/<model_pack>
#
# If checkpoint flags show no UMLS features needed (check inspect_medjex_checkpoint.py):
#   sbatch scripts/run_precompute_medjex_hyak.sh --no-umls
#
# The script checkpoints every 10 docs — safe to requeue if it times out.
#
#SBATCH --job-name=medjex-precompute
#SBATCH --account=stf-ckpt
#SBATCH --partition=ckpt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH --chdir=/mmfs1/home/pgarg2/2026-LING-573-Summarization-Project
#SBATCH --output=logs/medjex_precompute_%j.out
#SBATCH --error=logs/medjex_precompute_%j.err
#SBATCH --export=all
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=pgarg2@uw.edu

set -euo pipefail

export HF_HOME=/gscratch/scrubbed/pgarg2/hf-cache
export TRANSFORMERS_CACHE=/gscratch/scrubbed/pgarg2/hf-cache
export TORCH_HOME=/gscratch/scrubbed/pgarg2/torch-cache
export XDG_CACHE_HOME=/gscratch/scrubbed/pgarg2/xdg-cache
export MEDCAT_CACHE=/gscratch/scrubbed/pgarg2/medcat-cache

ENV_DIR=/gscratch/scrubbed/pgarg2/medjex-env
OUTPUT=/gscratch/scrubbed/pgarg2/medjex-spans/medjex_spans_test.json

mkdir -p logs "$(dirname "$OUTPUT")"

echo "=== MedJEx precompute started at $(date) ==="
echo "host: $(hostname)"
echo "args: $*"

source "$ENV_DIR/bin/activate"
echo "python: $(which python)"

python scripts/precompute_medjex_spans.py \
    --split test \
    --output "$OUTPUT" \
    --batch-size 16 \
    "$@"

# Copy final output into repo results dir so it's easy to commit/use
mkdir -p data/medjargone
cp "$OUTPUT" data/medjargone/medjex_spans_test.json
echo "Copied → data/medjargone/medjex_spans_test.json"

echo "=== Done at $(date) ==="
