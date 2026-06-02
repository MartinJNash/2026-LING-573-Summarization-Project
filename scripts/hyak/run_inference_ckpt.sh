#!/bin/bash
#SBATCH --job-name=medjargone-inference
#SBATCH --account=stf-ckpt
#SBATCH --partition=ckpt-all
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --chdir=/mmfs1/home/pgarg2/2026-LING-573-Summarization-Project
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --export=all
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=pgarg2@uw.edu

export HF_HOME=/gscratch/scrubbed/pgarg2/hf-cache
export TRANSFORMERS_CACHE=/gscratch/scrubbed/pgarg2/hf-cache
export NLTK_DATA=/gscratch/scrubbed/pgarg2/nltk-data

source /gscratch/scrubbed/pgarg2/medjargone/bin/activate

mkdir -p logs results/outputs

python run_inference.py \
    --model results/biobart-large \
    --split test \
    --batch-size 16 \
    --output results/outputs/biobart-large-finetuned.json
