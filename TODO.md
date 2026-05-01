# TODO

1. **Run eval across all models on the full test set** — all models were trained with `distilbert-base-uncased` for checkpoint selection (consistent across models); rerun eval with the fixed `eval_pipeline.py` (now uses `roberta-large` default) to get standard, comparable reported numbers; update `MODELS.md` with final results
