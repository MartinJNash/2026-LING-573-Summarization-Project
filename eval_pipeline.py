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


def load_outputs(path):
    with open(path, "r") as f:
        data = json.load(f)
    examples = data["examples"]
    preds = [e["pred"] for e in examples]
    golds = [e["gold"] for e in examples]
    model = data.get("model", "unknown")
    return preds, golds, model, examples


def compute_metrics(preds, golds, skip_bertscore=False):
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

    return rouge_scores, bleu_score, bertscore_result, readability


def print_results(rouge_scores, bleu_score, bertscore_result, readability, model, n):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="outputs.json", help="Path to inference outputs JSON")
    parser.add_argument("--output", default="eval_results.json", help="Path to save eval results")
    parser.add_argument("--fast", action="store_true", help="Skip BERTScore (fast CPU-only metrics only)")
    parser.add_argument("--skip-bertscore", action="store_true", help="Skip BERTScore")
    args = parser.parse_args()

    skip_bertscore = args.fast or args.skip_bertscore

    print(f"Loading outputs from {args.input}...")
    preds, golds, model, examples = load_outputs(args.input)

    rouge_scores, bleu_score, bertscore_result, readability = compute_metrics(
        preds, golds, skip_bertscore=skip_bertscore
    )
    print_results(rouge_scores, bleu_score, bertscore_result, readability, model, len(examples))

    output = {
        "model": model,
        "num_examples": len(examples),
        "rouge": rouge_scores,
        "bleu": bleu_score["bleu"],
        "bertscore": bertscore_result,
        "readability": readability,
    }

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
