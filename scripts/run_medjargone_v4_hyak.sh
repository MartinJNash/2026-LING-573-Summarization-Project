#!/bin/bash
#SBATCH --job-name=medjargone-v4
#SBATCH --account=stf
#SBATCH --partition=gpu-l40
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH --chdir=/mmfs1/home/pgarg2/2026-LING-573-Summarization-Project
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --export=all
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=pgarg2@uw.edu

export UV_CACHE_DIR=/gscratch/scrubbed/pgarg2/uv-cache
export HF_HOME=/gscratch/scrubbed/pgarg2/hf-cache
export TRANSFORMERS_CACHE=/gscratch/scrubbed/pgarg2/hf-cache

# UMLS_API_KEY must be exported in your shell before sbatch, or set here:
# export UMLS_API_KEY=<your_key>

source /gscratch/scrubbed/pgarg2/medjargone/bin/activate

mkdir -p logs results/outputs

PYTHONPATH=src python src/run_medjargone_v4.py \
    --split test \
    --output results/outputs/medjargone-v4-test.json \
    "$@"
