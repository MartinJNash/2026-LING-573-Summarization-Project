#!/bin/bash
#SBATCH --job-name=medjargone-v2-inference
#SBATCH --account=stf
#SBATCH --partition=ckpt-all
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH --chdir=/gscratch/scrubbed/<net-id>/2026-LING-573-Summarization-Project
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --export=all
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<net-id>@uw.edu

source ./environments/.envrc
source .venv/bin/activate

mkdir -p logs results/outputs

uv sync
uv run python -m spacy download en_core_web_sm # need to download spacy corpus for MLM
uv run python src/medjex/mlm_score_pipeline.py \
    --input biobart-large-lora-512.json \
    --output results/outputs/biobart-medjex_outputs.json

uv sync --group inference # need this to run vLLM
uv run python src/medjex/generate_rewrites.py \
    --model Qwen/Qwen2.5-3B-Instruct \
    --pipeline_file results/outputs/biobart-medjex_outputs.json \
    --output results/outputs/biobart-medjex-qwen_outputs.json