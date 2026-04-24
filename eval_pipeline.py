"""
Eval pipeline for MedJarGone summarization.
Reads from outputs.json produced by run_inference.py.
Computes ROUGE, BLEU, BERTScore, Flesch-Kincaid readability, SummaC faithfulness,
and medical concept overlap.

Usage:
    python eval_pipeline.py --input outputs/biobart-base.json
    python eval_pipeline.py --input outputs/biobart-base.json --output eval/biobart-base.json
"""

import json
import argparse
import evaluate
import bert_score
import textstat
from summac.model_summac import SummaCZS
import spacy


def load_outputs(path):
    with open(path, "r") as f:
        data = json.load(f)
    examples = data["examples"]
    preds = [e["pred"] for e in examples]
    golds = [e["gold"] for e in examples]
    sources = [e["input"] for e in examples]
    model = data.get("model", "unknown")
    return preds, golds, sources, model, examples


def compute_metrics(preds, golds, sources):
    # ROUGE — all variants; rougeLsum is the primary metric (matches MultiClinSum)
    print("Computing ROUGE...")
    rouge = evaluate.load("rouge")
    rouge_scores = rouge.compute(predictions=preds, references=golds, use_stemmer=True)

    # BLEU — reported for completeness
    print("Computing BLEU...")
    bleu = evaluate.load("bleu")
    bleu_score = bleu.compute(predictions=preds, references=[[g] for g in golds])

    # BERTScore — semantic similarity against reference; consistent model across all runs
    print("Computing BERTScore...")
    P, R, F1 = bert_score.score(
        preds, golds,
        lang="en",
        model_type="distilbert-base-uncased",
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

    # SummaC — NLI-based factual consistency; checks summary is supported by source
    # Expected: stable or slight decrease vs. gold (simplification may lose detail)
    print("Computing SummaC faithfulness...")
    summac_model = SummaCZS(granularity="sentence", model_name="vitc", device="cpu")
    summac_result = summac_model.score(sources, preds)
    faithfulness = {
        "summac_avg": sum(summac_result["scores"]) / len(summac_result["scores"])
    }

    # Medical concept overlap — scispacy biomedical NER F1 vs. gold
    # Approximates MEDCON without a UMLS license
    print("Computing medical concept overlap...")
    try:
        nlp = spacy.load("en_core_sci_sm")
        concept_f1s = []
        for pred, gold in zip(preds, golds):
            pred_concepts = {ent.text.lower() for ent in nlp(pred).ents}
            gold_concepts = {ent.text.lower() for ent in nlp(gold).ents}
            if not gold_concepts:
                continue
            precision = len(pred_concepts & gold_concepts) / len(pred_concepts) if pred_concepts else 0
            recall = len(pred_concepts & gold_concepts) / len(gold_concepts)
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            concept_f1s.append(f1)
        concept_overlap = {"concept_f1_avg": sum(concept_f1s) / len(concept_f1s) if concept_f1s else 0}
    except OSError:
        print("  scispacy model not found — skipping concept overlap.")
        print("  Install with: pip install scispacy && python -m spacy download en_core_sci_sm")
        concept_overlap = {"concept_f1_avg": None}

    return rouge_scores, bleu_score, bertscore_result, readability, faithfulness, concept_overlap


def print_results(rouge_scores, bleu_score, bertscore_result, readability, faithfulness, concept_overlap, model, n):
    print(f"\n========== EVAL RESULTS ==========")
    print(f"Model: {model} | Examples: {n}")
    print(f"Expected directions: ROUGE↓, BERTScore slight↓, FKGL significant↓, SummaC stable/slight↓")

    print(f"\nROUGE (expected ↓ vs. extractive baseline — patient-friendly summaries score lower):")
    for k, v in sorted(rouge_scores.items()):
        marker = " *" if k == "rougeLsum" else ""
        print(f"  {k}: {round(v * 100, 2)}{marker}")
    print(f"  (* primary metric)")

    print(f"\nBLEU: {round(bleu_score['bleu'] * 100, 2)}")

    print(f"\nBERTScore (expected slight ↓, more stable than ROUGE):")
    for k, v in bertscore_result.items():
        print(f"  {k}: {round(v, 4)}")

    print(f"\nFlesch-Kincaid Grade Level (expected pred < gold — lower = more readable):")
    print(f"  pred avg:  {round(readability['pred_fk_grade_avg'], 2)}")
    print(f"  gold avg:  {round(readability['gold_fk_grade_avg'], 2)}")
    delta = readability['pred_fk_grade_avg'] - readability['gold_fk_grade_avg']
    print(f"  delta:     {round(delta, 2)} ({'↓ more readable' if delta < 0 else '↑ less readable'})")

    print(f"\nSummaC Faithfulness (0–1, expected stable/slight ↓):")
    print(f"  avg: {round(faithfulness['summac_avg'], 4)}")

    print(f"\nMedical Concept Overlap F1 (scispacy NER, expected ↓ with readability gain):")
    val = concept_overlap["concept_f1_avg"]
    print(f"  avg: {round(val, 4) if val is not None else 'n/a'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="outputs.json", help="Path to inference outputs JSON")
    parser.add_argument("--output", default="eval_results.json", help="Path to save eval results")
    args = parser.parse_args()

    print(f"Loading outputs from {args.input}...")
    preds, golds, sources, model, examples = load_outputs(args.input)

    rouge_scores, bleu_score, bertscore_result, readability, faithfulness, concept_overlap = compute_metrics(preds, golds, sources)
    print_results(rouge_scores, bleu_score, bertscore_result, readability, faithfulness, concept_overlap, model, len(examples))

    output = {
        "model": model,
        "num_examples": len(examples),
        "rouge": rouge_scores,
        "bleu": bleu_score["bleu"],
        "bertscore": bertscore_result,
        "readability": readability,
        "faithfulness": faithfulness,
        "concept_overlap": concept_overlap,
    }

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
