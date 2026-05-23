# MedJarGone

Abstractive summarization of medical clinical notes for LING 573 (UW, Spring 2026), using the [MultiClinSum](https://zenodo.org/records/17341582) dataset.

**Baseline System (D2)**: See **[V1_MODELS.md](V1_MODELS.md)** for trained models, parameter counts, and evaluation results.

**Improved System (D3)**: Markdown file in progress...

## Setup

### Install uv (if not already installed)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc  # or restart your shell
```

### Install dependencies

1. Download the dataset from [Zenodo](https://zenodo.org/records/17341582) and place it under `data/`:
```bash
mkdir -p data && cd data
# Use -4 to force IPv4 (required on Patas/Dryas — IPv6 not supported)
wget -4 -O multiclinsum_gs_train_en.zip "https://zenodo.org/records/17341582/files/multiclinsum_gs_train_en.zip?download=1"
wget -4 -O multiclinsum_test_en.zip "https://zenodo.org/records/17341582/files/multiclinsum_test_en.zip?download=1"
unzip multiclinsum_gs_train_en.zip
unzip multiclinsum_test_en.zip
cd ..
```
2. Create and activate a virtual environment:
```bash
uv venv
source .venv/bin/activate
```
3. Install dependencies:
```bash
uv pip install -r requirements.txt
```

## Usage

**Train**
```bash
python train.py --base-model GanjinZero/biobart-v2-large --use-peft --output-dir results/biobart-large
```

**Inference**
```bash
python run_inference.py --model results/biobart-large --output results/outputs/biobart-large.json
```

**Evaluate**
```bash
python eval_pipeline.py --input results/outputs/biobart-large.json --output results/outputs/biobart-large-eval.json
```

Run `python <script> --help` for all options including `--num-epochs`, `--dataset`, `--batch-size`, `--num-examples`, `--fast`, `--skip-bertscore`.

## Cluster Setup

### Hyak / Klone (UW HPC — SLURM)

Hyak uses SLURM for job scheduling. Home directory quota is small — put venv and caches in `/gscratch/scrubbed/<netid>/`.

**First-time setup:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh && source ~/.bashrc

uv venv /gscratch/scrubbed/<netid>/medjargone --python 3.11
source /gscratch/scrubbed/<netid>/medjargone/bin/activate

export UV_CACHE_DIR=/gscratch/scrubbed/<netid>/uv-cache
export HF_HOME=/gscratch/scrubbed/<netid>/hf-cache

uv pip install -r requirements.txt
```

> Update `<netid>` and `--chdir` in SLURM scripts before submitting.

**Submit jobs:**
```bash
sbatch scripts/run_on_hyak.sh --base-model GanjinZero/biobart-v2-large --use-peft --output-dir results/biobart-large
sbatch scripts/run_inference_hyak.sh
sbatch scripts/run_eval_hyak.sh
```

**Monitor:**
```bash
squeue -u <netid>
tail -f logs/<jobid>.out
```

### Patas (UW Ling — Condor)

Patas uses HTCondor. Activate the conda env before running.

**Submit jobs:**
```bash
condor_submit scripts/train_patas.condor
condor_submit scripts/inference_patas.condor
```

**Run interactively:**
```bash
bash scripts/run_on_patas.sh --base-model GanjinZero/biobart-v2-large --use-peft --output-dir results/biobart-large
```

**Monitor:**
```bash
condor_q
```

> GPU nodes on Patas require `Requirements = (Machine == "patas-gn3.ling.washington.edu")` in the condor file (already set).
