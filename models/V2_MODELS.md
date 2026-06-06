# Model Registry

Fine-tuned on [MultiClinSum](https://zenodo.org/records/17341582) GS training split (594 examples, 90/10 train/val, seed=42) using LoRA (r=16, α=16, q_proj + v_proj, dropout=0.1). Checkpoints selected by best BERTScore F1 on validation set.

## Models

| Model | HF Hub | Base | Params (total / LoRA) | Training |
|---|---|---|---|---|
| BioBART-v2-large LoRA | `Pika4028/biobart-v2-large-multiclinsum-lora` | `GanjinZero/biobart-v2-large` | 406M / ~2.36M (0.58%) | 10 epochs |
| BioBART-v2-large + MedJEx + Qwen2.5 | `` | `GanjinZero/biobart-v2-base` | _same as above_ | _same as above_ |
| Qwen2.5 | `Qwen/Qwen2.5-3B-Instruct` | — | 3B / — | no fine-tuning |

> BioBART-v2-large estimated from architecture (d_model=1024, 36 adapter matrices).

## Evaluation Results

Computed with `eval_pipeline.py`. Primary metric: **rougeLsum** (matches MultiClinSum shared task).
FK Grade: lower pred vs. gold = more patient-readable.
SARI Score: [0, 100] -> higher score = higher correlation with human judgment

### Full test set (3,396 examples)

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L | rougeLsum | BLEU | BERTScore F1 | FK pred | FK gold | SARI |
|---|---|---|---|---|---|---|---|---|---|
| BioBART-v2-large LoRA | 36.68 | 16.54 | 24.79 | 24.83 | 12.25 | 85.47 | 13.05 | 14.58 | 36.53 |
| BioBART + MedJEx + Qwen | 33.92 | 12.86 | 21.99 | 22.17 | 8.98 | 85.08 | 11.92 | 14.58 | 30.63 |
| Qwen2.5 | 33.45 | 9.22 | 19.34 | 21.75 | 5.19 | 85.36 | 11.61 | 14.58 | 40.53 |
> Run eval: `python src/eval_pipeline.py --input results/v2/v2_outputs/<model>.json --output results/v2/v2_eval/<model>-eval.json`

### LLM-as-Judge evaluation (full test set, 3,396 examples)

Scored by GPT-4o on informativeness, simplification quality, and faithfulness (each 0–100).

| System | Informativeness | Simplification | Faithfulness |
|---|---|---|---|
| BioBART-large LoRA (D2) | 88.67 | 84.13 | 96.01 |
| BioBART + MedJEx + Qwen (D3) | 91.65 | 87.48 | 97.04 |
| Qwen2.5 zero-shot | 91.23 | 86.74 | 96.21 |

## Inference Outputs

All outputs in `results/v2/v2_outputs/`.

| Model | Path | Examples | Format |
|---|---|---|---|
| BioBART-v2-large LoRA | `results/v2/v2_outputs/biobart-large-lora-512.json` | 3,396 | JSON |
| BioBART + MedJEx + Qwen | `results/v2/v2_outputs/biobart-medjex-qwen_outputs.json` | 3,396 | JSON |
| Qwen2.5 | `results/v2/v2_outputs/qwen_outputs.json` | 3,396 | JSON |
