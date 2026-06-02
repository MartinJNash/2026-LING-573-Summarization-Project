"""
Compute BERTScore + SARI for D4 prelim vs Qwen zero-shot vs BioBART-large-lora-512.
Aligns all systems to the 75 examples present in the D4 prelim file.

Usage:
    uv run python scripts/compute_bertscore_sari.py
    uv run python scripts/compute_bertscore_sari.py --output results/outputs/prelim_bertscore_sari.json
"""

from __future__ import annotations

import os

# Compatibility shim: huggingface_hub 1.x removed is_offline_mode from public API
# but transformers 5.x still imports it at module load time.
import huggingface_hub as _hf_hub
if not hasattr(_hf_hub, "is_offline_mode"):
    _hf_hub.is_offline_mode = lambda: os.environ.get("HF_HUB_OFFLINE", "0") == "1"

import argparse
import json
from pathlib import Path

import bert_score
from easse.sari import corpus_sari


def load_by_id(path: str | Path) -> dict[str, dict]:
    with open(path) as f:
        data = json.load(f)
    exs = data["examples"] if isinstance(data, dict) else data
    return {str(e["id"]): e for e in exs}


def compute_for(label: str, preds: list[str], golds: list[str],
                sources: list[str]) -> dict:
    print(f"  BERTScore ({label})…", flush=True)
    _, _, bs_f1 = bert_score.score(preds, golds, lang="en", verbose=False)
    bs = round(bs_f1.mean().item(), 4)

    print(f"  SARI ({label})…", flush=True)
    sari = round(corpus_sari(orig_sents=sources, sys_sents=preds,
                              refs_sents=[golds]), 2)

    print(f"  {label}: BERTScore F1={bs}  SARI={sari}")
    return {"bertscore_f1": bs, "sari": sari, "n": len(preds)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/outputs/prelim_bertscore_sari.json")
    args = parser.parse_args()

    root = Path(__file__).parents[1]

    v4   = load_by_id(root / "results/outputs/medjargone-v4-prelim.json")
    bl   = load_by_id(root / "results/outputs/biobart-large-lora-512.json")
    qwen = load_by_id(root / "results/v2_results/v2_outputs/qwen_outputs.json")

    ids = list(v4.keys())
    print(f"Evaluating on {len(ids)} examples (D4 prelim IDs)\n")

    def extract(store, ids):
        rows = [store[i] for i in ids if i in store]
        return ([r["pred"] for r in rows],
                [r["gold"] for r in rows],
                [r["input"] for r in rows])

    results = {}
    for label, store in [("D4-medjargone-v4", v4),
                          ("BioBART-large-lora-512", bl),
                          ("Qwen2.5-3B-zero-shot", qwen)]:
        preds, golds, sources = extract(store, ids)
        results[label] = compute_for(label, preds, golds, sources)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
