# Model Registry

Fine-tuned on [MultiClinSum](https://zenodo.org/records/17341582) GS training split (594 examples, 90/10 train/val, seed=42) using LoRA (r=16, α=16, q_proj + v_proj, dropout=0.1). Checkpoints selected by best BERTScore F1 on validation set.

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

### Full test set (3,396 examples)

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L | rougeLsum | BLEU | BERTScore F1 | FK pred | FK gold |
|---|---|---|---|---|---|---|---|---|
| BART-base baseline | 34.88 | 15.11 | 22.67 | 22.71 | 10.57 | 84.63 | 12.24 | 14.58 |
| BART-base LoRA | 36.11 | 15.88 | 23.24 | 23.29 | 11.02 | 85.04 | 12.32 | 14.58 |
| BioBART-v2-base baseline | 35.32 | 15.28 | 22.85 | 22.89 | 10.60 | 84.49 | 12.71 | 14.58 |
| BioBART-v2-base LoRA | 35.85 | 15.49 | 23.08 | 23.12 | 10.85 | 84.73 | 12.75 | 14.58 |
| BioBART-v2-large LoRA | 37.03 | 16.46 | 24.92 | 24.97 | 13.00 | 85.44 | 13.01 | 14.58 |

> Run eval: `python src/eval_pipeline.py --input results/v1/<model>/<model>.json --output results/v1/<model>/<model>-eval.json`

## Inference Outputs

All outputs in `results/v1/`. Each model has its own subdirectory containing the predictions JSON and eval results.

| Model | Path | Examples | Format |
|---|---|---|---|
| BART-base baseline | `results/v1/bart-base-baseline/bart-base-baseline.json` | 3,396 | JSON |
| BART-base LoRA | `results/v1/bart-base-lora/bart-base-lora.json` | 3,396 | JSON |
| BioBART-v2-base baseline | `results/v1/biobart-base-baseline/biobart-base-baseline.json` | 3,396 | JSON |
| BioBART-v2-base LoRA | `results/v1/biobart-base-lora/biobart-base-lora.json` | 3,396 | JSON |
| BioBART-v2-large LoRA | `results/v1/biobart-large-lora/biobart-large-lora.json` | 3,396 | JSON |
