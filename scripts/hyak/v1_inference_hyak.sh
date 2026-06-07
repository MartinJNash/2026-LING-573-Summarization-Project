#!/bin/bash
#SBATCH --job-name=medjargone-inference
#SBATCH --account=stf
#SBATCH --partition=gpu-2080ti
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --chdir=/gscratch/scrubbed/<net-id>/2026-LING-573-Summarization-Project
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --export=all
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<net-id>@uw.edu

source environments/.envrc
source .venv/bin/activate

mkdir -p logs results/outputs

uv run python src/finetune/run_inference.py "$@"
