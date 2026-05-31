#!/bin/bash
# Job array — splits 3396 test examples across 16 tasks (213 examples each).
# Max 8 concurrent to stay within stf fairshare limits (Hyak best practice).
#
# Submit:  sbatch scripts/run_medjargone_v4_array_hyak.sh
# Merge:   python scripts/merge_v4_chunks.py
#
#SBATCH --job-name=medjargone-v4-array
#SBATCH --account=stf
#SBATCH --partition=gpu-l40
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH --array=0-15%8
#SBATCH --chdir=/mmfs1/home/pgarg2/2026-LING-573-Summarization-Project
#SBATCH --output=logs/v4array_%A_%a.out
#SBATCH --error=logs/v4array_%A_%a.err
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

echo "Waiting for Ollama server (task ${SLURM_ARRAY_TASK_ID})..."
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

mkdir -p logs results/outputs/chunks

CHUNK_SIZE=213
START=$(( SLURM_ARRAY_TASK_ID * CHUNK_SIZE ))

echo "Array task ${SLURM_ARRAY_TASK_ID}: examples ${START}–$((START + CHUNK_SIZE - 1))"

PYTHONPATH=src python src/run_medjargone_v4.py \
    --split test \
    --start-idx "$START" \
    --num-examples "$CHUNK_SIZE" \
    --output "results/outputs/chunks/medjargone-v4-test-chunk-${SLURM_ARRAY_TASK_ID}.json"

kill "$OLLAMA_PID" 2>/dev/null || true
