# MedJarGone

Abstractive summarization of medical clinical notes into plain language, built for LING 573 (UW, Spring 2026) using the [MultiClinSum](https://zenodo.org/records/17341582) dataset.

Three systems were developed across deliverables D2–D4:

| Deliverable | System | Approach | Best SARI |
|---|---|---|---|
| D2 | BART/BioBART LoRA | Fine-tuned seq2seq (LoRA) | — |
| D3 | BioBART + MedJEx + Qwen | LoRA + jargon detection + LLM rewrite | 30.63 |
| D4 | MedJarGone v4 | UMLS-grounded glossary + Qwen2.5 zero-shot | 39.80 |

Full results: [`models/V1_MODELS.md`](models/V1_MODELS.md), [`models/V2_MODELS.md`](models/V2_MODELS.md), [`models/V4_MODELS.md`](models/V4_MODELS.md).

---

## Repository layout

```
data/                    # dataset (not tracked — download separately, see below)
  multiclinsum_gs_train_en/
  multiclinsum_test_en/
environments/
  requirements.txt       # pinned dependencies (use with uv)
models/                  # model registry docs + HuggingFace links
results/
  v1/                    # D2 inference outputs + eval
  v2/                    # D3 inference outputs + eval
  v4/                    # D4 (MedJarGone v4) outputs
  llm_eval/              # LLM-as-judge cross-system comparison
scripts/
  hyak/                  # SLURM scripts for Hyak (UW HPC)
  patas/                 # HTCondor scripts for Patas (UW Ling)
  *.py                   # standalone utility scripts
src/
  medjargone/            # D4 pipeline package (UMLS lookup, rewrite, verify)
  medjex/                # MedJEx jargon-detection submodule
  eval_pipeline.py       # shared evaluation script (ROUGE, BLEU, BERTScore, SARI)
  model.py               # shared LoRA model wrapper
  train.py               # D2/D3 fine-tuning entry point
  run_inference.py       # D2/D3 inference entry point
  run_medjargone_v4.py   # D4 batch inference entry point
```

---

## Setup

### Prerequisites

- Python 3.11
- [`uv`](https://github.com/astral-sh/uv) package manager

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc   # or restart your shell
```

### Install dependencies

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install -r environments/requirements.txt
```

On Linux (Hyak/Patas), `uv` automatically installs the CUDA 12.4 build of PyTorch instead of CPU-only.

### Download the dataset

```bash
mkdir -p data && cd data
# Use -4 to force IPv4 (required on Patas/Dryas — IPv6 not supported)
wget -4 -O multiclinsum_gs_train_en.zip "https://zenodo.org/records/17341582/files/multiclinsum_gs_train_en.zip?download=1"
wget -4 -O multiclinsum_test_en.zip "https://zenodo.org/records/17341582/files/multiclinsum_test_en.zip?download=1"
unzip multiclinsum_gs_train_en.zip
unzip multiclinsum_test_en.zip
cd ..
```

---

## D2 — BART/BioBART LoRA fine-tuning

Fine-tunes BART-base or BioBART-v2 with LoRA on the MultiClinSum training split (594 examples).

**Train:**
```bash
python src/train.py --base-model GanjinZero/biobart-v2-large --use-peft --output-dir results/biobart-large
```

**Inference:**
```bash
python src/run_inference.py --model results/biobart-large --output results/v1/biobart-large/biobart-large.json
```

**Evaluate:**
```bash
python src/eval_pipeline.py --input results/v1/biobart-large/biobart-large.json \
    --output results/v1/biobart-large/biobart-large-eval.json
```

Run `python src/<script>.py --help` for all options (`--num-epochs`, `--batch-size`, `--num-examples`, `--skip-bertscore`, etc.).

See [models/V1_MODELS.md](models/V1_MODELS.md) for trained checkpoints and full evaluation results.

---

## D3 — BioBART + MedJEx + Qwen rewrite pipeline

Extends D2 with MedJEx jargon detection and a Qwen2.5 rewrite step. The BioBART-v2-large LoRA checkpoint is already on HuggingFace; inference runs directly from the hub.

See [models/V2_MODELS.md](models/V2_MODELS.md) for trained checkpoints and full evaluation results.

---

## D4 — MedJarGone v4 (UMLS glossary + Qwen2.5 zero-shot)

Rule-based pipeline that rewrites clinical notes into plain language using a UMLS-grounded medical glossary and Qwen2.5 as the language model. No fine-tuning required.

**Pipeline stages:** preprocess → MedJEx jargon extraction → UMLS glossary lookup → Qwen2.5 rewrite → MiniCheck factual verification.

See [models/V4_MODELS.md](models/V4_MODELS.md) for architecture details and evaluation results.

### Prerequisites (one-time)

Get a free UMLS API key at [uts.nlm.nih.gov](https://uts.nlm.nih.gov) and export it:

```bash
export UMLS_API_KEY=<your_key>
```

### Run locally

```bash
python src/run_medjargone_v4.py --split test --output results/v4/medjargone-v4-test.json
```

### Run on Hyak

Qwen2.5 inference requires a GPU node. Full setup and job scripts are in `scripts/hyak/`.

```bash
# Pre-warm the UMLS cache (CPU, ~1 hr — do this once)
sbatch scripts/hyak/prewarm_umls_cache_hyak.sh

# Pre-compute MedJEx jargon spans (GPU, separate conda env)
sbatch scripts/hyak/precompute_medjex_hyak.sh

# Run full test set via job array
sbatch scripts/hyak/medjargone_v4_array_hyak.sh
python scripts/merge_v4_chunks.py
```

---

## Cluster Setup

### Hyak / Klone (UW HPC — SLURM)

Home directory quota is small — put your venv and caches in `/gscratch/scrubbed/<netid>/`.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh && source ~/.bashrc

uv venv /gscratch/scrubbed/<netid>/medjargone --python 3.11
source /gscratch/scrubbed/<netid>/medjargone/bin/activate

export UV_CACHE_DIR=/gscratch/scrubbed/<netid>/uv-cache
export HF_HOME=/gscratch/scrubbed/<netid>/hf-cache

uv pip install -r environments/requirements.txt
```

Update `<netid>` and `--chdir` in any SLURM script before submitting. Monitor jobs with:

```bash
squeue -u <netid>
tail -f logs/<jobid>.out
```

### Patas (UW Ling — HTCondor)

Condor scripts are in `scripts/patas/`. Run setup once before submitting:

```bash
bash scripts/patas/initial_setup_patas.sh
```

Submit jobs:

```bash
condor_submit scripts/patas/train_patas.cmd
condor_submit scripts/patas/inference_patas.cmd
condor_submit scripts/patas/eval_patas.cmd
```

Monitor with `condor_q`.

> GPU nodes on Patas require `Requirements = (Machine == "patas-gn3.ling.washington.edu")` in the condor file (already set).
