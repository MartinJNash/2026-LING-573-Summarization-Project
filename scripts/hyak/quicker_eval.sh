#!/bin/bash
#SBATCH --job-name=medjargone-llm-eval
#SBATCH --account=stf
#SBATCH --partition=gpu-2080ti
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gpus=1
#SBATCH --time=08:00:00
#SBATCH --chdir=/gscratch/scrubbed/srigor/medjargone
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --export=all
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=srigor@uw.edu

# Squid proxy intercepts 127.0.0.1 requests — exclude it so the ollama
# Python client can reach the local ollama server directly
export no_proxy="${no_proxy},127.0.0.1,localhost"
export NO_PROXY="${NO_PROXY},127.0.0.1,localhost"

# Ollama model storage — keeps models out of the 10GB home dir
export OLLAMA_MODELS=/gscratch/scrubbed/srigor/ollama/models
mkdir -p "$OLLAMA_MODELS"


module load apptainer
#apptainer cache clean
#export APPTAINER_CACHEDIR=/tmp
#apptainer build --fakeroot ollama.sif ollama.def

OLLAMA_SIF=/mmfs1/gscratch/scrubbed/srigor/medjargone/ollama.sif

# Start the Ollama server in the background via the prebuilt container
apptainer exec --nv \
    --bind /gscratch/ \
    --env OLLAMA_MODELS="$OLLAMA_MODELS" \
    "$OLLAMA_SIF" ollama serve &
OLLAMA_PID=$!

# Wait for server to be ready
echo "Waiting for Ollama server..."
for i in $(seq 3 60); do
    curl -sf http://localhost:11434 > /dev/null 2>&1 && break
    sleep 2
done

# Pull the model (no-op if already cached in OLLAMA_MODELS)
apptainer exec --nv \
    --bind /gscratch/ \
    --env OLLAMA_MODELS="$OLLAMA_MODELS" \
    "$OLLAMA_SIF" ollama pull llama3.1:8b-instruct-q2_K # using a WAY faster model now

source /mmfs1/home/srigor/medjargone/.envrc
source .venv/bin/activate

mkdir -p quicker_eval

uv sync
uv run python llm_eval_hyak.py --model llama3.1:8b-instruct-q2_K --input results/qwen_outputs.json --output quicker_eval/qwen.json
uv run python llm_eval_hyak.py --model llama3.1:8b-instruct-q2_K --input results/biobart-large-lora-512.json --output quicker_eval/d2_baseline_eval.json
uv run python llm_eval_hyak.py --model llama3.1:8b-instruct-q2_K --input results/biobart-medjex-qwen_outputs.json --output quicker_eval/d3_medjex_eval.json
