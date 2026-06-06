#!/bin/bash
# Preliminary run — 75 examples on ckpt partition.
# Used to validate pipeline before full array job.
#
# Submit:  sbatch scripts/hyak/medjargone_v4_prelim_hyak.sh
#
#SBATCH --job-name=medjargone-v4-prelim
#SBATCH --account=stf-ckpt
#SBATCH --partition=ckpt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --time=01:30:00
#SBATCH --chdir=/mmfs1/home/pgarg2/2026-LING-573-Summarization-Project
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --export=all
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=pgarg2@uw.edu

export UV_CACHE_DIR=/gscratch/scrubbed/pgarg2/uv-cache
export HF_HOME=/gscratch/scrubbed/pgarg2/hf-cache
export HF_DATASETS_CACHE=/gscratch/scrubbed/pgarg2/hf-cache/datasets
export TRANSFORMERS_CACHE=/gscratch/scrubbed/pgarg2/hf-cache
export TORCH_HOME=/gscratch/scrubbed/pgarg2/torch-cache
export XDG_CACHE_HOME=/gscratch/scrubbed/pgarg2/xdg-cache
export APPTAINER_CACHEDIR=/gscratch/scrubbed/pgarg2/apptainer-cache
export no_proxy="${no_proxy},127.0.0.1"
export NO_PROXY="${NO_PROXY},127.0.0.1"
export OLLAMA_MODELS=/gscratch/scrubbed/pgarg2/ollama-models
mkdir -p "$OLLAMA_MODELS"

OLLAMA_SIF=/mmfs1/sw/containers/ollama/ollama.sif

apptainer exec --nv \
    --bind /gscratch/ \
    --env OLLAMA_MODELS="$OLLAMA_MODELS" \
    "$OLLAMA_SIF" ollama serve &
OLLAMA_PID=$!

echo "Waiting for Ollama server..."
for i in $(seq 1 30); do
    curl -sf http://localhost:11434 > /dev/null 2>&1 && break
    sleep 2
done

apptainer exec --nv \
    --bind /gscratch/ \
    --env OLLAMA_MODELS="$OLLAMA_MODELS" \
    "$OLLAMA_SIF" ollama pull qwen2.5:3b

source /gscratch/scrubbed/pgarg2/medjargone/bin/activate
uv pip install ollama

mkdir -p logs results/outputs

# 75 examples ≈ 59 min at observed ~47 s/example; 90-min wall is intentional buffer
PYTHONPATH=src python src/run_medjargone_v4.py \
    --split test \
    --num-examples 75 \
    --output results/outputs/medjargone-v4-prelim.json \
    "$@"

kill "$OLLAMA_PID" 2>/dev/null || true
