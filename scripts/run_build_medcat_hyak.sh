#!/bin/bash
# Build MedCAT model pack from UMLS MRCONSO.RRF — CPU-only, ~20-30 min.
#
#   sbatch scripts/run_build_medcat_hyak.sh
#
#SBATCH --job-name=build-medcat-cdb
#SBATCH --account=stf-ckpt
#SBATCH --partition=ckpt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --chdir=/mmfs1/home/pgarg2/2026-LING-573-Summarization-Project
#SBATCH --output=logs/build_medcat_%j.out
#SBATCH --error=logs/build_medcat_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=pgarg2@uw.edu

set -euo pipefail

ENV_DIR=/gscratch/scrubbed/pgarg2/medjex-env

mkdir -p logs

echo "=== build_medcat_cdb started at $(date) ==="
echo "host: $(hostname)"

eval "$(conda shell.bash hook)"
conda activate "$ENV_DIR"
echo "python: $(which python)"

python scripts/build_medcat_cdb.py

echo "=== Done at $(date) ==="
