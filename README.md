# MedJarGone

Abstractive summarization of medical clinical notes for LING 573 (UW, Spring 2026), using the [MultiClinSum](https://doi.org/10.5281/zenodo.10813550) dataset.

## Setup

1. Download the dataset from [Zenodo](https://doi.org/10.5281/zenodo.10813550) and place it under `data/`
2. Create and activate a virtual environment:
```bash
uv venv --python 3.11
source .venv/bin/activate
```
3. Install dependencies:
```bash
uv pip install -r requirements.txt
```

## Usage

**Train**
```bash
uv run python pipeline.py --base-model GanjinZero/biobart-v2-large --use-peft --output-dir results/biobart-large
```

**Inference**
```bash
uv run python run_inference.py --model results/biobart-large --output outputs/biobart-large.json
```

**Evaluate**
```bash
uv run python eval_pipeline.py --input outputs/biobart-large.json --output eval/biobart-large.json
```

Run `uv run python <script> --help` for all options.

## Cluster (Hyak)

### First-time setup

Storage on Hyak: the home directory has a small quota. Put the venv and all caches in `/gscratch/scrubbed/<netid>/` (large, but files are auto-deleted after 21 days).

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc

# Create venv in scrubbed
uv venv /gscratch/scrubbed/<netid>/medjargone --python 3.11
source /gscratch/scrubbed/<netid>/medjargone/bin/activate

# Point caches to scrubbed so home quota is not exceeded
export UV_CACHE_DIR=/gscratch/scrubbed/<netid>/uv-cache
export HF_HOME=/gscratch/scrubbed/<netid>/hf-cache

# Install dependencies
uv pip install -r requirements.txt
```

### Submitting a training job

Before submitting, update `NETID` and `--chdir` in `scripts/run_on_hyak.sh`, then:

```bash
sbatch scripts/run_on_hyak.sh --base-model GanjinZero/biobart-v2-large --use-peft --output-dir results/biobart-large
```

Monitor the job:
```bash
squeue -u <netid>
tail -f logs/<jobid>.out
```
