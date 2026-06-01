"""
M7 — Safety-focused evaluation: D2 / Qwen / D3 / v4.

Reads pre-generated output JSON files (one per system) and produces:
  1. Standard metrics table  — ROUGE-1/2, BLEU, BERTScore F1, FK grade, SARI
  2. Safety table            — per-example flags from the Stage-4 verifier applied
                               to every system's output

The verifier is run on existing predictions, so it does NOT require the LLM
to be loaded — only scispaCy + MiniCheck + facts from the v4 outputs.

Usage:
    python -m medjargone.eval.run_eval \
        --d2   results/outputs/biobart-large-lora.json \
        --qwen results/outputs/qwen-zero-shot.json \
        --d3   results/outputs/d3-pipeline.json \
        --v4   results/outputs/v4-pipeline.json \
        --output eval/v4_safety_table.csv

Each input JSON must follow the schema:
    {"model": "...", "examples": [{"id": ..., "input": ..., "gold": ..., "pred": ...}]}
v4 JSON may additionally carry "facts" and "verification" per example.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# ── Metric imports (same as existing eval_pipeline.py) ───────────────────────
import evaluate
import bert_score
import textstat

try:
    from easse.sari import corpus_sari as _easse_sari
    _SARI_OK = True
except ImportError:
    _SARI_OK = False


# ── Safety check (runs Stage-4 Tier 1 rules only — no LLM needed) ─────────────
from medjargone.pipeline.verify import (
    rule_check, minicheck_verify, VerificationReport
)


# ── I/O helpers ───────────────────────────────────────────────────────────────

def load_output_json(path: Path) -> tuple[str, list[dict]]:
    with open(path) as fh:
        data = json.load(fh)
    model = data.get("model", path.stem)
    return model, data["examples"]


# ── Standard metrics ──────────────────────────────────────────────────────────

def compute_standard_metrics(
    preds: list[str],
    golds: list[str],
    sources: Optional[list[str]] = None,
) -> dict:
    rouge = evaluate.load("rouge")
    bleu  = evaluate.load("bleu")

    r = rouge.compute(predictions=preds, references=golds)
    b = bleu.compute(predictions=preds,
                     references=[[g] for g in golds])

    _, _, bs_f1 = bert_score.score(preds, golds, lang="en", verbose=False)

    fk_grades = [textstat.flesch_kincaid_grade(p) for p in preds]
    avg_fk = sum(fk_grades) / len(fk_grades) if fk_grades else 0.0

    out = {
        "rouge1":       round(r["rouge1"] * 100, 2),
        "rouge2":       round(r["rouge2"] * 100, 2),
        "bleu":         round(b["bleu"]   * 100, 2),
        "bertscore_f1": round(bs_f1.mean().item(), 4),
        "fk_grade":     round(avg_fk, 2),
    }

    if _SARI_OK and sources:
        sari = _easse_sari(orig_sents=sources, sys_sents=preds, refs_sents=[golds])
        out["sari"] = round(sari, 2)
    else:
        out["sari"] = None

    return out


# ── Safety metrics ────────────────────────────────────────────────────────────

def _safety_flags_from_verification(v: dict) -> dict:
    """Extract safety flag counts from a v4 verification dict."""
    return {
        "missing_numbers":    int(bool(v.get("missing_numbers"))),
        "wrong_laterality":   int(v.get("wrong_laterality", False)),
        "wrong_organs":       int(bool(v.get("wrong_organs"))),
        "missing_coverage":   int(bool(v.get("missing_coverage"))),
        "unsupported_claims": len(v.get("unsupported_claims", [])),
        "revised":            int(v.get("revised", False)),
    }


def compute_safety_metrics(
    examples: list[dict],
    system_tag: str,
) -> list[dict]:
    """
    Run Tier-1 + Tier-2 (MiniCheck) checks on the predictions.
    For v4 examples that already carry a 'verification' dict, those are reused.
    Returns a list of per-example dicts.
    """
    rows = []
    for ex in examples:
        pred   = ex.get("pred", "")
        source = ex.get("input", "")

        # Use pre-computed verification from v4 run if available
        if "verification" in ex:
            flags = _safety_flags_from_verification(ex["verification"])
        elif "facts" in ex:
            facts  = ex["facts"]
            report = rule_check(facts, pred)
            unsupported, _ = minicheck_verify(pred, facts, source)
            flags = {
                "missing_numbers":    int(bool(report.missing_numbers)),
                "wrong_laterality":   int(report.wrong_laterality),
                "wrong_organs":       int(bool(report.wrong_organs)),
                "missing_coverage":   int(bool(report.missing_coverage)),
                "unsupported_claims": len(unsupported),
                "revised":            0,
            }
        else:
            # For D2/Qwen/D3 without facts: run MiniCheck only (no Tier-1 facts)
            unsupported, _ = minicheck_verify(pred, {}, source)
            flags = {
                "missing_numbers":    None,
                "wrong_laterality":   None,
                "wrong_organs":       None,
                "missing_coverage":   None,
                "unsupported_claims": len(unsupported),
                "revised":            0,
            }

        rows.append({
            "system": system_tag,
            "id":     ex.get("id", ""),
            **flags,
        })
    return rows


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MedJarGone v4 safety evaluation")
    parser.add_argument("--d2",     type=Path, default=None)
    parser.add_argument("--qwen",   type=Path, default=None)
    parser.add_argument("--d3",     type=Path, default=None)
    parser.add_argument("--v4",     type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("eval/v4_safety_table.csv"))
    parser.add_argument("--metrics-output", type=Path,
                        default=Path("eval/v4_standard_metrics.json"))
    parser.add_argument("--ids", type=Path, default=None,
                        help="JSON file with list of integer IDs to evaluate (for subset comparison)")
    args = parser.parse_args()

    # Optional ID filter — lets us compare all systems on the same prelim subset
    id_filter: Optional[set] = None
    if args.ids:
        with open(args.ids) as fh:
            id_filter = {str(i) for i in json.load(fh)}

    systems = {
        k: v for k, v in [
            ("D2",   args.d2),
            ("Qwen", args.qwen),
            ("D3",   args.d3),
            ("v4",   args.v4),
        ] if v is not None
    }

    if not systems:
        print("No input files specified. Use --d2, --qwen, --d3, --v4.")
        sys.exit(1)

    all_metrics = {}
    all_safety_rows: list[dict] = []

    for tag, path in systems.items():
        print(f"\n=== {tag}: {path} ===")
        model_name, examples = load_output_json(path)

        if id_filter:
            examples = [e for e in examples if str(e.get("id", "")) in id_filter]
            print(f"  Filtered to {len(examples)} examples matching --ids")

        preds   = [e.get("pred", "") for e in examples]
        golds   = [e.get("gold",  "") for e in examples]
        sources = [e.get("input", "") for e in examples]

        print(f"  {len(examples)} examples  |  model: {model_name}")

        print("  Computing standard metrics…")
        metrics = compute_standard_metrics(preds, golds, sources)
        all_metrics[tag] = metrics
        print(f"  ROUGE-1={metrics['rouge1']}  ROUGE-2={metrics['rouge2']}  "
              f"BLEU={metrics['bleu']}  BSF1={metrics['bertscore_f1']}  "
              f"FK={metrics['fk_grade']}  SARI={metrics['sari']}")

        print("  Computing safety flags…")
        safety_rows = compute_safety_metrics(examples, tag)
        all_safety_rows.extend(safety_rows)

    # ── Standard metrics JSON ─────────────────────────────────────────────────
    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.metrics_output, "w") as fh:
        json.dump(all_metrics, fh, indent=2)
    print(f"\nStandard metrics → {args.metrics_output}")

    # ── Safety table CSV ──────────────────────────────────────────────────────
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "system", "id",
        "missing_numbers", "wrong_laterality", "wrong_organs",
        "missing_coverage", "unsupported_claims", "revised",
    ]
    with open(args.output, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_safety_rows)
    print(f"Safety table     → {args.output}")

    # ── Summary safety counts ─────────────────────────────────────────────────
    print("\n=== Safety summary (counts over all examples) ===")
    header = f"{'System':<8}  {'NoBadNums':>9}  {'BadLat':>6}  {'BadOrg':>6}  "
    header += f"{'NoCovg':>6}  {'Unsupport':>9}"
    print(header)
    for tag in systems:
        rows = [r for r in all_safety_rows if r["system"] == tag]
        def _sum(key):
            vals = [r[key] for r in rows if r[key] is not None]
            return sum(vals) if vals else "N/A"
        print(
            f"{tag:<8}  {_sum('missing_numbers'):>9}  {_sum('wrong_laterality'):>6}  "
            f"{_sum('wrong_organs'):>6}  {_sum('missing_coverage'):>6}  "
            f"{_sum('unsupported_claims'):>9}"
        )


if __name__ == "__main__":
    main()
