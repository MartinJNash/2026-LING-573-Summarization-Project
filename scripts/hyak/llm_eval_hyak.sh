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
#SBATCH --chdir=/gscratch/scrubbed/<net-id>/2026-LING-573-Summarization-Project
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --export=all
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<net-id>@uw.edu

# Squid proxy intercepts 127.0.0.1 requests — exclude it so the ollama
# Python client can reach the local ollama server directly
export no_proxy="${no_proxy},127.0.0.1,localhost"
export NO_PROXY="${NO_PROXY},127.0.0.1,localhost"

# Ollama storage — keeps models/cache out of the 10GB home dir
#   you can comment or uncomment out these lines as needed!!!
# rm -rf ~/.ollama # remove ollama dir from home directory on hyak
# mkdir /gscratch/scrubbed/<net-id>/.ollama # make ollama dir in temp storage
# ln -s /gscratch/scrubbed/<net-id>/.ollama ~/.ollama # make a symlink to the temp storage
export OLLAMA_CACHE=/gscratch/scrubbed/<net-id>/.ollama/cache
export OLLAMA_MODELS=/gscratch/scrubbed/<net-id>/.ollama/models
mkdir -p "$OLLAMA_MODELS" "$OLLAMA_CACHE"

# the ollama.sif on hyak is outdated and doesn't allow pydantic json schemas. need to build from scratch!
OLLAMA_DEF=/gscratch/scrubbed<net-id>/2026-LING-573-Summarization-Project/src/llm_eval/ollama.def
OLLAMA_SIF=/gscratch/scrubbed/<net-id>/2026-LING-573-Summarization-Project/src/llm_eval/ollama.sif
module load apptainer
export APPTAINER_CACHEDIR=/tmp
apptainer build --fakeroot "$OLLAMA_SIF" "$OLLAMA_DEF"

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
    "$OLLAMA_SIF" ollama pull llama3.1:8b-instruct-q2_K
    # make sure to use a model that ISN'T the same as the zero-shot/rewrite model!
    # using quantized version for time + memory efficiency

source environments/.envrc
source .venv/bin/activate

uv sync
uv run python src/llm_eval/llm_eval_hyak.py "$@"
