#!/bin/bash
# CPU-only job — no GPU needed for scispaCy NER + UTS API calls.
# Run this before the array inference job to pre-populate uts_cache.sqlite.
# Once complete, every UMLS lookup in the array tasks is a local SQLite read.
#
# Submit:  sbatch scripts/hyak/prewarm_umls_cache_hyak.sh
#
#SBATCH --job-name=medjargone-prewarm
#SBATCH --account=stf
#SBATCH --partition=cpu-g2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --chdir=/mmfs1/home/<netid>/2026-LING-573-Summarization-Project
#SBATCH --output=logs/%j-prewarm.out
#SBATCH --error=logs/%j-prewarm.err
#SBATCH --export=all
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<netid>@uw.edu

export UV_CACHE_DIR=/gscratch/scrubbed/<netid>/uv-cache
export HF_HOME=/gscratch/scrubbed/<netid>/hf-cache
export no_proxy="${no_proxy},127.0.0.1"
export NO_PROXY="${NO_PROXY},127.0.0.1"

# UMLS_API_KEY must be set in your shell before sbatch, or uncomment below:
# export UMLS_API_KEY=<your_key>

source /gscratch/scrubbed/<netid>/medjargone/bin/activate

mkdir -p logs data/medjargone/cache

echo "Starting UMLS cache pre-warm for test split…"
PYTHONPATH=src python scripts/prewarm_umls_cache.py --split test

echo ""
echo "Cache written to data/medjargone/cache/uts_cache.sqlite"
echo "Array inference job will now use cached lookups (no live API calls)."
