#!/bin/bash
# Quick smoke test: 5-example inference + eval with GPU/memory diagnostics.
# Run interactively on a GPU node before submitting the full Slurm job:
#
#   srun --account=stf --partition=gpu-l40 --gpus=1 --mem=32G --time=0:30:00 --pty bash
#   cd 2026-LING-573-Summarization-Project
#   bash scripts/test_run.sh

set -e

export UV_CACHE_DIR=/gscratch/scrubbed/pgarg2/uv-cache
export HF_HOME=/gscratch/scrubbed/pgarg2/hf-cache
export TRANSFORMERS_CACHE=/gscratch/scrubbed/pgarg2/hf-cache

source /gscratch/scrubbed/pgarg2/medjargone/bin/activate

TEST_OUTPUT=results/outputs/test-smoke.json
TEST_EVAL=results/outputs/test-smoke-eval.json

mkdir -p logs results/outputs

echo "========================================"
echo "GPU info:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo "========================================"

echo ""
echo ">>> Running inference (5 examples)..."
time python run_inference.py \
    --model results/biobart-large \
    --split test \
    --num-examples 5 \
    --batch-size 4 \
    --output "$TEST_OUTPUT"

echo ""
echo "GPU memory after inference:"
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader

echo ""
echo ">>> Running eval..."
time python eval_pipeline.py \
    --input "$TEST_OUTPUT" \
    --output "$TEST_EVAL"

echo ""
echo "========================================"
echo "Smoke test complete. Check $TEST_EVAL for metric sanity."
echo "If inference < 2 min and eval < 5 min, the full job should be well within the 2h wall time."
echo "========================================"
