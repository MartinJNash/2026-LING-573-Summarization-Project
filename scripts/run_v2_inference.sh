#!/bin/bash
#SBATCH --job-name=medjargone-v2
#SBATCH --account=stf
#SBATCH --partition=ckpt-all
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH --chdir=/gscratch/scrubbed/srigor/medjargone
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --export=all
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=srigor@uw.edu

source /mmfs1/home/srigor/medjargone/.envrc
source /gscratch/scrubbed/srigor/medjargone/bin/activate

mkdir -p logs results/outputs

uv sync
uv run python -m spacy download en_core_web_sm # need to download spacy corpus for MLM
uv run python mlm_score_pipeline.py \
    --input biobart-large-lora-512.json \
    --output ./results/outputs/biobart-medjex_outputs.json

uv sync --group inference
uv run python generate_rewrites.py \
    --model Qwen/Qwen2.5-3B-Instruct --pipeline_file ./results/outputs/biobart-medjex_outputs.json  --output ./results/outputs/biobart-medjex-qwen_outputs.json

uv lock