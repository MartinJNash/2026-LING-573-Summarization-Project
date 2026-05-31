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
export HF_DATASETS_CACHE=/gscratch/scrubbed/pgarg2/hf-cache/datasets
export TRANSFORMERS_CACHE=/gscratch/scrubbed/pgarg2/hf-cache
export TORCH_HOME=/gscratch/scrubbed/pgarg2/torch-cache
export XDG_CACHE_HOME=/gscratch/scrubbed/pgarg2/xdg-cache
export APPTAINER_CACHEDIR=/gscratch/scrubbed/pgarg2/apptainer-cache

# Squid proxy intercepts 127.0.0.1 requests — exclude it so the ollama
# Python client can reach the local ollama server directly
export no_proxy="${no_proxy},127.0.0.1"
export NO_PROXY="${NO_PROXY},127.0.0.1"

# Ollama model storage — keeps models out of the 10GB home dir
export OLLAMA_MODELS=/gscratch/scrubbed/pgarg2/ollama-models
mkdir -p "$OLLAMA_MODELS"

# UMLS_API_KEY must be exported in your shell before sbatch, or set here:
# export UMLS_API_KEY=<your_key>

OLLAMA_SIF=/mmfs1/sw/containers/ollama/ollama.sif

# Start the Ollama server in the background via the prebuilt container
apptainer exec --nv \
    --bind /gscratch/ \
    --env OLLAMA_MODELS="$OLLAMA_MODELS" \
    "$OLLAMA_SIF" ollama serve &
OLLAMA_PID=$!

# Wait for server to be ready
echo "Waiting for Ollama server..."
for i in $(seq 1 30); do
    curl -sf http://localhost:11434 > /dev/null 2>&1 && break
    sleep 2
done

# Pull the model (no-op if already cached in OLLAMA_MODELS)
apptainer exec --nv \
    --bind /gscratch/ \
    --env OLLAMA_MODELS="$OLLAMA_MODELS" \
    "$OLLAMA_SIF" ollama pull qwen2.5:3b

source /gscratch/scrubbed/pgarg2/medjargone/bin/activate

# Install ollama Python client if not already present
uv pip install ollama

mkdir -p logs results/outputs

PYTHONPATH=src python src/run_medjargone_v4.py \
    --split test \
    --output results/outputs/medjargone-v4-test.json \
    "$@"

kill "$OLLAMA_PID" 2>/dev/null || true
