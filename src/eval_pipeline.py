"""
Compatibility entry point for `python -m src.eval_pipeline`.
Reads from outputs.jsonl + meta.json produced by src/run_inference.py.
Computes BLEU, ROUGE, BERTScore, and Flesch-Kincaid readability.

For the richer eval with SummaC and medical concept overlap, use eval_pipeline.py directly.

Usage:
    python -m src.eval_pipeline --input outputs/outputs.jsonl
"""

import json
import argparse
from pathlib import Path
import evaluate
import bert_score
import textstat


def load_outputs(path):
    path = Path(path)
    with open(path, "r") as f:
        examples = [json.loads(line) for line in f if line.strip()]
    preds = [e["pred"] for e in examples]
    golds = [e["gold"] for e in examples]

    meta_path = path.parent / "meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            model = json.load(f).get("model", "unknown")
    else:
        model = "unknown"

    return preds, golds, model, examples


def compute_metrics(preds, golds):
    # --- ROUGE ---
    print("Computing ROUGE...")
    rouge = evaluate.load("rouge")
    rouge_scores = rouge.compute(predictions=preds, references=golds, use_stemmer=True)

    # --- BLEU ---
    print("Computing BLEU...")
    bleu = evaluate.load("bleu")
    bleu_score = bleu.compute(predictions=preds, references=[[g] for g in golds])

    # --- BERTScore ---
    print("Computing BERTScore...")
    P, R, F1 = bert_score.score(preds, golds, lang="en", verbose=False)
    bertscore_result = {
        "precision": P.mean().item(),
        "recall": R.mean().item(),
        "f1": F1.mean().item()
    }

    # --- Readability (Flesch-Kincaid) ---
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
    for k, v in rouge_scores.items():
        print(f"  {k}: {round(v * 100, 2)}")

    print(f"\nBLEU: {round(bleu_score['bleu'] * 100, 2)}")

    print(f"\nBERTScore:")
    for k, v in bertscore_result.items():
        print(f"  {k}: {round(v, 4)}")

    print(f"\nFlesch-Kincaid Grade Level:")
    print(f"  pred avg: {round(readability['pred_fk_grade_avg'], 2)}")
    print(f"  gold avg: {round(readability['gold_fk_grade_avg'], 2)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="outputs/outputs.jsonl", help="Path to inference outputs JSONL")
    parser.add_argument("--output", default="eval_results.json", help="Path to save eval results")
    args = parser.parse_args()

    print(f"Loading outputs from {args.input}...")
    preds, golds, model, examples = load_outputs(args.input)

    rouge_scores, bleu_score, bertscore_result, readability = compute_metrics(preds, golds)
    print_results(rouge_scores, bleu_score, bertscore_result, readability, model, len(examples))

    output = {
        "model": model,
        "num_examples": len(examples),
        "rouge": rouge_scores,
        "bleu": bleu_score["bleu"],
        "bertscore": bertscore_result,
        "readability": readability,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
