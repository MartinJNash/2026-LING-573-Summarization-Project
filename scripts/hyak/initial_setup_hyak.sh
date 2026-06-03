#!/bin/bash
#SBATCH --job-name=medjargone-setup
#SBATCH --account=stf
#SBATCH --partition=ckpt-all
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH --chdir=/gscratch/scrubbed/<net-id>
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --export=all
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<net-id>@uw.edu

# ensure caches are in temporary storage (NOT user home directory)
mkdir -p uv-cache hf-cache torch-cache

# install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc

# clone GitHub repository
git clone https://github.com/MartinJNash/2026-LING-573-Summarization-Project.git
cd 2026-LING-573-Summarization-Project

# install datasets
mkdir -p data && cd data
wget -4 -O multiclinsum_gs_train_en.zip "https://zenodo.org/records/17341582/files/multiclinsum_gs_train_en.zip?download=1"
wget -4 -O multiclinsum_test_en.zip "https://zenodo.org/records/17341582/files/multiclinsum_test_en.zip?download=1"
unzip multiclinsum_gs_train_en.zip
unzip multiclinsum_test_en.zip
rm *.zip # delete zip files after expansion
cd ..

# activate Python environment and install dependencies from pyproject.toml
uv venv --python 3.12 # must be within versions 3.11 and 3.13
source .venv/bin/activate
uv pip install -r environments/requirements.txt
uv sync
uv lock
