# D4 System: MedJarGone v4

Rule-based pipeline that rewrites clinical notes into plain language using a UMLS-grounded medical glossary and Qwen2.5 as the language model. No fine-tuning — the system runs zero-shot.

## Architecture

1. **Preprocess** — split source into sentences, normalize whitespace
2. **Extract facts** — identify key clinical entities (MedJEx jargon spans + scispaCy NER)
3. **Glossary lookup** — resolve jargon terms against UMLS via the UTS REST API; cache results in SQLite
4. **Rewrite** — prompt Qwen2.5 with the source + glossary to produce a plain-language summary
5. **Verify** — run MiniCheck factual-consistency check; flag low-confidence outputs

## Models

| Component | Model | Notes |
|---|---|---|
| LLM rewriter | `Qwen/Qwen2.5-3B-Instruct` | Zero-shot; served via Ollama on Hyak |
| Jargon detector | MedJEx checkpoint | Pre-run; spans cached per document |
| Factual verifier | MiniCheck (`lytang/MiniCheck-Flan-T5-Large`) | Optional post-hoc check |

## Evaluation Results (N=75, test split)

All systems scored on the same 75-example subset from the full test set.

### Automatic metrics

| System | ROUGE-1 | ROUGE-2 | ROUGE-L | BLEU | BERTScore F1 | FK Grade | SARI |
|---|---|---|---|---|---|---|---|
| D2 System (BioBART-large LoRA) | 34.55 | 14.57 | 22.80 | 8.80 | 85.00 | 12.47 | 36.17 |
| **D4 System (MedJarGone v4)** | **34.80** | **11.59** | **21.63** | **6.60** | **85.60** | **13.03** | **39.80** |
| Qwen2.5 zero-shot | 33.20 | 9.30 | 19.20 | 2.80 | 85.20 | 11.86 | 40.20 |

### LLM-as-Judge evaluation

Scored by GPT-4o on informativeness, simplification quality, and faithfulness (each 0–100).

| System | Informativeness | Simplification | Faithfulness |
|---|---|---|---|
| D2 System (BioBART-large LoRA) | 90.00 | 87.01 | 96.36 |
| D3 System (BioBART + MedJEx + Qwen) | 91.28 | 87.76 | 97.11 |
| **D4 System (MedJarGone v4)** | **83.03** | **77.99** | **92.16** |
| Qwen2.5 zero-shot | 91.15 | 85.84 | 96.31 |

> Full test-set run in progress. Outputs at `results/v4/`.

## Running

```bash
export UMLS_API_KEY=<your_key>
python run_medjargone_v4.py --split test --output results/v4/medjargone-v4-test.json
```

On Hyak (GPU required for Qwen2.5 via Ollama):
```bash
sbatch scripts/hyak/medjargone_v4_hyak.sh
# or job array for full test set:
sbatch scripts/hyak/medjargone_v4_array_hyak.sh
python scripts/merge_v4_chunks.py
```

### Setup (one-time, before running)

```bash
# Pre-warm the UMLS cache (CPU-only, ~1hr)
sbatch scripts/hyak/prewarm_umls_cache_hyak.sh

# Pre-compute MedJEx jargon spans (GPU, separate conda env — see setup_medjex_env_hyak.sh)
sbatch scripts/hyak/precompute_medjex_hyak.sh
```
