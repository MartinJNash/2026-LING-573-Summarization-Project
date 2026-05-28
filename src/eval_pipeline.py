"""
Eval pipeline for MedJarGone summarization.
Reads from outputs.json produced by run_inference.py.
Computes ROUGE, BLEU, BERTScore, Flesch-Kincaid readability,
and medical concept overlap.

Usage:
    python eval_pipeline.py --input outputs/biobart-base.json
    python eval_pipeline.py --input outputs/biobart-base.json --output eval/biobart-base.json
"""

import os
import json
import argparse
import evaluate
import bert_score
import textstat

try:
    from easse.sari import corpus_sari as _easse_corpus_sari
    _EASSE_AVAILABLE = True
except ImportError:
    _EASSE_AVAILABLE = False


def load_outputs(path):
    # Support both JSON ({"examples": [...]}) and JSONL (one record per line)
    with open(path, "r") as f:
        content = f.read()
    try:
        data = json.loads(content)
        examples = data["examples"]
        model = data.get("model", "unknown")
    except (json.JSONDecodeError, KeyError):
        examples = [json.loads(line) for line in content.splitlines() if line.strip()]
        model = "unknown"

    preds = [e["pred"] for e in examples]
    golds = [e["gold"] for e in examples]
    sources = [e["input"] for e in examples] if examples and "input" in examples[0] else None
    return preds, golds, sources, model, examples


def compute_sari(sources, preds, golds):
    """
    Compute SARI score using easse. Returns None if easse is not installed
    or sources are unavailable.
    refs_sents shape: (n_references, n_samples) — wrap single reference list.
    """
    if not _EASSE_AVAILABLE:
        print("Skipping SARI (easse not installed).")
        return None
    if sources is None:
        print("Skipping SARI (no source/input text in outputs).")
        return None
    print("Computing SARI...")
    score = _easse_corpus_sari(
        orig_sents=sources,
        sys_sents=preds,
        refs_sents=[golds],
    )
    return score


def compute_metrics(preds, golds, sources=None, skip_bertscore=False):
    # Use PID as experiment_id to avoid temp file collisions across parallel jobs
    experiment_id = str(os.getpid())

    # ROUGE — all variants; rougeLsum is the primary metric (matches MultiClinSum)
    print("Computing ROUGE...")
    rouge = evaluate.load("rouge", experiment_id=experiment_id)
    rouge_scores = rouge.compute(predictions=preds, references=golds, use_stemmer=True)

    # BLEU — reported for completeness
    print("Computing BLEU...")
    bleu = evaluate.load("bleu", experiment_id=experiment_id)
    bleu_score = bleu.compute(predictions=preds, references=[[g] for g in golds])

    # BERTScore — semantic similarity against reference
    if skip_bertscore:
        print("Skipping BERTScore.")
        bertscore_result = None
    else:
        print("Computing BERTScore...")
        P, R, F1 = bert_score.score(
            preds, golds,
            lang="en",
            verbose=False,
        )
        bertscore_result = {
            "precision": P.mean().item(),
            "recall": R.mean().item(),
            "f1": F1.mean().item(),
        }

    # Flesch-Kincaid Grade Level — lower pred score = more readable for patients
    print("Computing readability...")
    fk_preds = [textstat.flesch_kincaid_grade(p) for p in preds]
    fk_golds = [textstat.flesch_kincaid_grade(g) for g in golds]
    readability = {
        "pred_fk_grade_avg": sum(fk_preds) / len(fk_preds),
        "gold_fk_grade_avg": sum(fk_golds) / len(fk_golds),
    }

    sari_score = compute_sari(sources, preds, golds)

    return rouge_scores, bleu_score, bertscore_result, readability, sari_score


def print_results(rouge_scores, bleu_score, bertscore_result, readability, sari_score, model, n):
    print(f"\n========== EVAL RESULTS ==========")
    print(f"Model: {model} | Examples: {n}")

    print(f"\nROUGE:")
    for k, v in sorted(rouge_scores.items()):
        marker = " *" if k == "rougeLsum" else ""
        print(f"  {k}: {round(v * 100, 2)}{marker}")
    print(f"  (* primary metric)")

    print(f"\nBLEU: {round(bleu_score['bleu'] * 100, 2)}")

    print(f"\nBERTScore:")
    if bertscore_result is not None:
        for k, v in bertscore_result.items():
            print(f"  {k}: {round(v, 4)}")
    else:
        print(f"  skipped")

    print(f"\nFlesch-Kincaid Grade Level (lower pred = more readable for patients):")
    print(f"  pred avg:  {round(readability['pred_fk_grade_avg'], 2)}")
    print(f"  gold avg:  {round(readability['gold_fk_grade_avg'], 2)}")
    delta = readability['pred_fk_grade_avg'] - readability['gold_fk_grade_avg']
    print(f"  delta:     {round(delta, 2)} ({'↓ more readable' if delta < 0 else '↑ less readable'})")

    print(f"\nSARI:")
    if sari_score is not None:
        print(f"  {round(sari_score, 4)}")
    else:
        print(f"  skipped (requires source/input text in outputs)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="outputs.json", help="Path to inference outputs JSON or JSONL")
    parser.add_argument("--output", default="eval_results.json", help="Path to save eval results")
    parser.add_argument("--fast", action="store_true", help="Skip BERTScore (fast CPU-only metrics only)")
    parser.add_argument("--skip-bertscore", action="store_true", help="Skip BERTScore")
    parser.add_argument("--skip-sari", action="store_true", help="Skip SARI even if source text is available")
    args = parser.parse_args()

    skip_bertscore = args.fast or args.skip_bertscore

    print(f"Loading outputs from {args.input}...")
    preds, golds, sources, model, examples = load_outputs(args.input)

    if args.skip_sari:
        sources = None

    rouge_scores, bleu_score, bertscore_result, readability, sari_score = compute_metrics(
        preds, golds, sources=sources, skip_bertscore=skip_bertscore
    )
    print_results(rouge_scores, bleu_score, bertscore_result, readability, sari_score, model, len(examples))

    output = {
        "model": model,
        "num_examples": len(examples),
        "rouge": rouge_scores,
        "bleu": bleu_score["bleu"],
        "bertscore": bertscore_result,
        "readability": readability,
        "sari": sari_score,
    }

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
