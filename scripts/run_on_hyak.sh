#!/bin/bash
#SBATCH --job-name=medjargone-train
#SBATCH --account=stf
#SBATCH --partition=gpu-2080ti
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --chdir=/mmfs1/home/pgarg2/2026-LING-573-Summarization-Project
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --export=all
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=pgarg2@uw.edu

export UV_CACHE_DIR=/gscratch/scrubbed/pgarg2/uv-cache
export HF_HOME=/gscratch/scrubbed/pgarg2/hf-cache
export TRANSFORMERS_CACHE=/gscratch/scrubbed/pgarg2/hf-cache

source /gscratch/scrubbed/pgarg2/medjargone/bin/activate

mkdir -p logs

python pipeline.py "$@"
