#!/bin/bash
#SBATCH --job-name=medjargone-evaluation
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

mkdir -p logs results/outputs v2_eval

uv sync 
uv run python eval_pipeline.py --input biobart-large-lora-512.json --output biobart-large-lora-512-eval.json
uv run python eval_pipeline.py --input results/outputs/biobart-medjex-qwen_outputs.json --output biobart-medjex-qwen-eval.json
uv run python eval_pipeline.py --input results/outputs/qwen_outputs.json --output qwen-eval.json

mv ./*eval.json v2_eval
