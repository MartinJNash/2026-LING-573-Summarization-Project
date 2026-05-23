#!/bin/bash
#SBATCH --job-name=medjargone-llm-inference
#SBATCH --account=stf
#SBATCH --partition=ckpt-all
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=11G
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

uv sync --group inference
uv run python llm_summarizer.py \
    --model Qwen/Qwen2.5-3B-Instruct \
    --split test \
    --output results/outputs/llm_only_outputs.json
