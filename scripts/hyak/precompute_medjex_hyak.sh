#!/bin/bash
# Precompute MedJEx jargon spans for the full MultiClinSum test split on Hyak.
#
# First run setup_medjex_env_hyak.sh once, then:
#   sbatch scripts/hyak/precompute_medjex_hyak.sh \
#       --medcat-path /gscratch/scrubbed/<netid>/medcat/<model_pack>
#
# If the MedJEx checkpoint uses no UMLS features (Binary_flag=False, TF_flag=False,
# MLM_flag=False), pass --no-umls to skip MedCAT:
#   sbatch scripts/hyak/precompute_medjex_hyak.sh --no-umls
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
#SBATCH --chdir=/mmfs1/home/<netid>/2026-LING-573-Summarization-Project
#SBATCH --output=logs/medjex_precompute_%j.out
#SBATCH --error=logs/medjex_precompute_%j.err
#SBATCH --export=all
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<netid>@uw.edu

set -euo pipefail

export HF_HOME=/gscratch/scrubbed/<netid>/hf-cache
export TRANSFORMERS_CACHE=/gscratch/scrubbed/<netid>/hf-cache
export TORCH_HOME=/gscratch/scrubbed/<netid>/torch-cache
export XDG_CACHE_HOME=/gscratch/scrubbed/<netid>/xdg-cache
export MEDCAT_CACHE=/gscratch/scrubbed/<netid>/medcat-cache

ENV_DIR=/gscratch/scrubbed/<netid>/medjex-env
OUTPUT=/gscratch/scrubbed/<netid>/medjex-spans/medjex_spans_test.json

mkdir -p logs "$(dirname "$OUTPUT")"

echo "=== MedJEx precompute started at $(date) ==="
echo "host: $(hostname)"
echo "args: $*"

eval "$(conda shell.bash hook)"
conda activate "$ENV_DIR"
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
