# MedJarGone

Abstractive summarization of medical clinical notes, with a focus on producing patient-friendly output by reducing medical jargon.

Built for LING 573 (University of Washington, Spring 2026) using the [MultiClinSum](https://doi.org/10.5281/zenodo.10813550) shared task dataset.

---

## Setup

### 1. Get the data

Download the MultiClinSum dataset from [Zenodo](https://doi.org/10.5281/zenodo.10813550) and place it under `data/` so the structure looks like:

```
data/
├── multiclinsum_gs_train_en/
│   ├── fulltext/
│   └── summaries/
├── multiclinsum_large-scale_train_en/
│   ├── fulltext/
│   └── summaries/
└── multiclinsum_test_en/
    ├── fulltext/
    └── summaries/
```

### 2. Create the environment

```bash
conda env create -f environment.yml
conda activate medjargone
```

> **GPU note:** The default `environment.yml` installs CPU PyTorch. On a GPU cluster, install the appropriate CUDA-enabled PyTorch wheel instead:
> `pip install torch --index-url https://download.pytorch.org/whl/cu118`

---

## Running the Pipeline

### Step 1 — Train

Fine-tune a model on the gold-standard training split:

```bash
python pipeline.py \
    --base-model GanjinZero/biobart-v2-base \
    --use-peft \
    --output-dir results/biobart-base-lora
```

**Arguments:**
- `--base-model` — HuggingFace model ID or local path. Supported options:
  - `GanjinZero/biobart-v2-base` *(recommended baseline)*
  - `GanjinZero/biobart-v2-large`
  - `facebook/bart-base`
  - `facebook/bart-large`
- `--use-peft` — Enable LoRA fine-tuning (recommended; reduces GPU memory)
- `--output-dir` — Where to save the best checkpoint

Checkpoints are saved to `--output-dir` during training. The best checkpoint is selected by BERTScore F1 on the validation set.

### Step 2 — Inference

Run the trained model on the test set and save predictions:

```bash
python run_inference.py \
    --model results/biobart-base-lora \
    --split test \
    --output outputs/biobart-base-lora.json
```

**Arguments:**
- `--model` — Path to trained model directory (or HuggingFace model ID for zero-shot)
- `--split` — `test` (default) or `train`
- `--num-examples` — Limit to N examples (default: all)
- `--output` — Output JSON file path

### Step 3 — Evaluate

Compute evaluation metrics on the inference outputs:

```bash
python eval_pipeline.py \
    --input outputs/biobart-base-lora.json \
    --output eval/biobart-base-lora-results.json
```

**Arguments:**
- `--input` — JSON file produced by `run_inference.py`
- `--output` — Where to save the metrics JSON

**Metrics reported:**
- ROUGE-1/2/L (lexical overlap; expected to decrease for patient-friendly rewording)
- BERTScore F1 (semantic similarity; expected to remain stable)
- Flesch-Kincaid Grade Level (readability; expected to decrease toward grade 6-7)

---

## Running on a Cluster

See `scripts/` for job submission files.

**Patas (HTCondor):**
```bash
condor_submit scripts/run_on_patas.condor.bat
```
This submits 4 parallel jobs: BioBART-base, BioBART-large, BART-base, BART-large — each with LoRA.

---

## Project Structure

```
├── pipeline.py          # Training script (fine-tunes model on MultiClinSum)
├── run_inference.py     # Inference script (generates summaries, saves to JSON)
├── eval_pipeline.py     # Evaluation script (computes metrics on inference output)
├── model.py             # Summarizer class (loads tokenizer + model)
├── read_data.py         # Data loading utilities for MultiClinSum
├── environment.yml      # Conda environment
├── scripts/
│   ├── run_on_patas.sh           # Shell wrapper for Patas
│   └── run_on_patas.condor.bat   # HTCondor batch submission
├── data/                # MultiClinSum dataset (not tracked in git)
├── models/              # Pretrained model checkpoints (tracked via git-lfs)
└── results/             # Training outputs (not tracked in git)
```
