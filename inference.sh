#!/bin/sh

# Set up environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate medjargone-gpu

# Run code
python src/run_inference.py "$@"
