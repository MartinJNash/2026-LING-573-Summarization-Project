#!/bin/bash

mkdir -p ~/uv-cache ~/hf-cache ~/torch-cache

export UV_CACHE_DIR=~/uv-cache
export TRANSFORMERS_CACHE=~/hf-cache
export TORCH_HOME=~/torch-cache

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

# activate Python environment and install dependencies
uv venv --python 3.12 # must be within versions 3.11 and 3.13
source .venv/bin/activate
uv pip install -r environments/requirements.txt
uv sync
uv lock