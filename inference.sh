#!/bin/bash

echo "=== inference.sh started at $(date) ==="
echo "host: $(hostname)"
echo "cwd:  $(pwd)"
echo "args: $@"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate medjargone-gpu

echo "python: $(which python)"
echo "conda env: $CONDA_DEFAULT_ENV"

python -m src.run_inference "$@"
echo "=== inference.sh finished at $(date) ==="
