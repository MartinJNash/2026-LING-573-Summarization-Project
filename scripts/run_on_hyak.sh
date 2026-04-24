#!/bin/bash
#SBATCH --job-name=medjargone-train
#SBATCH --account=stf
#SBATCH --partition=gpu-2080ti
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=pgarg2@uw.edu

source /gscratch/scrubbed/pgarg2/medjargone/bin/activate

mkdir -p logs

python pipeline.py "$@"
