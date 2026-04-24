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
> **GPU:** Replace the torch install with a CUDA-enabled wheel:
> `uv pip install torch --index-url https://download.pytorch.org/whl/cu118`

## Usage

**Train**
```bash
uv run python pipeline.py --base-model GanjinZero/biobart-v2-base --use-peft --output-dir results/biobart-base
```

**Inference**
```bash
uv run python run_inference.py --model results/biobart-base --output outputs/biobart-base.json
```

**Evaluate**
```bash
uv run python eval_pipeline.py --input outputs/biobart-base.json --output eval/biobart-base.json
```

Run `uv run python <script> --help` for all options.

## Cluster

See `scripts/` for Patas (HTCondor) and Hyak (SLURM) job submission files.
