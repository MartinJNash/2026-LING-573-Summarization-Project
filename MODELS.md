# Model Registry

Fine-tuned on [MultiClinSum](https://doi.org/10.5281/zenodo.10813550) GS training split (594 examples, 90/10 train/val, seed=42) using LoRA (r=16, α=16, q_proj + v_proj, dropout=0.1). Checkpoints selected by best BERTScore F1 on validation set.

## Models

| Model | HF Hub | Base | Params (total / LoRA) | Training |
|---|---|---|---|---|
| BART-base baseline | `facebook/bart-base` | — | 139M / — | no fine-tuning |
| BART-base LoRA | `mjnash-uw/bart-base-lora` | `facebook/bart-base` | 139M / ~885K (0.63%) | 3 epochs |
| BioBART-v2-base baseline | `GanjinZero/biobart-v2-base` | — | 139M / — | no fine-tuning |
| BioBART-v2-base LoRA | `mjnash-uw/biobart-base-lora` | `GanjinZero/biobart-v2-base` | 139M / ~885K (0.63%) | 3 epochs |
| BioBART-v2-large LoRA | `Pika4028/biobart-v2-large-multiclinsum-lora` | `GanjinZero/biobart-v2-large` | 406M / ~2.36M (0.58%) | 10 epochs |

> BART-base LoRA param count from training logs. BioBART-v2-large estimated from architecture (d_model=1024, 36 adapter matrices).

## Evaluation Results

Computed with `eval_pipeline.py`. Primary metric: **rougeLsum** (matches MultiClinSum shared task).
FK grade: lower pred vs. gold = more patient-readable.

### 339 examples (10% of test set, seed=42)

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L | rougeLsum | BLEU | BERTScore F1 | FK pred | FK gold |
|---|---|---|---|---|---|---|---|---|
| BART-base baseline | 35.15 | 15.09 | 22.70 | 22.77 | 10.51 | 84.67 | 12.24 | 14.70 |
| BART-base LoRA | 36.46 | 15.88 | 23.47 | 23.47 | 10.93 | 85.15 | 12.35 | 14.70 |
| BioBART-v2-base baseline | 35.80 | 15.44 | 23.06 | 23.13 | 10.64 | 84.54 | 12.65 | 14.70 |
| BioBART-v2-base LoRA | 36.27 | 15.70 | 23.35 | 23.41 | 10.95 | 84.80 | 12.67 | 14.70 |

### Full test set (3,396 examples)

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L | rougeLsum | BLEU | BERTScore F1 | FK pred | FK gold |
|---|---|---|---|---|---|---|---|---|
| BioBART-v2-large LoRA | *pending* | | | | | | | |

> Run eval: `python eval_pipeline.py --input results/outputs/biobart-large-lora.json --output results/outputs/biobart-large-lora-eval.json`

## Inference Outputs

All outputs in `results/outputs/`. JSONL format includes a `meta.json` alongside; JSON format is self-contained.

| Model | Path | Examples | Format |
|---|---|---|---|
| BART-base baseline | `results/outputs/bart-base-baseline/` | 339 | JSONL |
| BART-base LoRA | `results/outputs/bart-base-lora/` | 339 | JSONL |
| BioBART-v2-base baseline | `results/outputs/biobart-base-baseline/` | 339 | JSONL |
| BioBART-v2-base LoRA | `results/outputs/biobart-base-lora/` | 339 | JSONL |
| BioBART-v2-large LoRA | `results/outputs/biobart-large-lora.json` | 3,396 | JSON |
