"""
MedJarGone v4 pipeline orchestrator.

Runs Stages 1 → 4 for a single case report and returns a structured result dict.

Usage (CLI):
    python -m medjargone.pipeline.run --model Qwen/Qwen2.5-3B-Instruct \
        --input path/to/report.txt

Usage (library):
    from medjargone.pipeline.run import run_pipeline, load_llm
    llm_fn = load_llm("Qwen/Qwen2.5-3B-Instruct")
    result = run_pipeline(source_text, llm_fn)
    print(result["summary"])

Environment:
    export UMLS_API_KEY=...   # required for UMLS lookup
                             # omit only if local UMLS subset (umls_subset.sqlite) is built
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Callable, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from medjargone import config
from medjargone.pipeline.preprocess import preprocess_clinical_text
from medjargone.pipeline.extract_facts import extract_facts
from medjargone.pipeline.glossary import (
    build_glossary, UMLSIndex, UTSClient, APICache
)
from medjargone.pipeline.rewrite import generate_rewrite
from medjargone.pipeline.verify import verify_and_fix


# ── LLM wrapper ───────────────────────────────────────────────────────────────

def load_llm(model_name: str = config.LLM_MODEL) -> Callable[[str], str]:
    """
    Returns llm_fn(prompt: str) -> str backed by Ollama.
    Requires `ollama serve` to be running (started by the Slurm script).
    """
    import ollama as _ollama

    def llm_fn(prompt: str) -> str:
        response = _ollama.chat(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a precise medical assistant."},
                {"role": "user",   "content": prompt},
            ],
            options={
                "temperature": 0.2,
                "top_p": 0.9,
                "num_predict": config.LLM_MAX_NEW_TOKENS,
            },
        )
        return response["message"]["content"]

    return llm_fn


# ── Single-document pipeline ──────────────────────────────────────────────────

def run_pipeline(
    source_text: str,
    llm_fn: Callable[[str], str],
    umls_index: Optional[UMLSIndex] = None,
    uts_client: Optional[UTSClient] = None,
    cache: Optional[APICache]       = None,
) -> dict:
    """
    Run all four stages for one case report.

    Returns:
    {
      "summary": str,
      "facts": dict,
      "glossary": list[dict],
      "verification": {
          "missing_numbers", "wrong_laterality", "wrong_organs", "wrong_drugs",
          "missing_coverage", "unsupported_claims", "gray_zone_claims",
          "revised", "summary_str"
      },
      "stage_times_ms": {"s1": int, "s2": int, "s3": int, "s4": int}
    }
    """
    times: dict[str, int] = {}

    # medspaCy preprocessing — section detection + ConText infrastructure
    t = time.monotonic()
    preprocessed = preprocess_clinical_text(source_text)
    times["s0_preprocess"] = int((time.monotonic() - t) * 1000)

    # Stage 1 — fact extraction (uses section-tagged text when available)
    t = time.monotonic()
    facts = extract_facts(source_text, llm_fn, preprocessed=preprocessed)
    times["s1"] = int((time.monotonic() - t) * 1000)

    # Stage 2 — UMLS-grounded glossary (ConText filtering via preprocessed)
    t = time.monotonic()
    glossary = build_glossary(
        source_text,
        preprocessed=preprocessed,
        umls_index=umls_index,
        uts_client=uts_client,
        cache=cache,
    )
    times["s2"] = int((time.monotonic() - t) * 1000)

    # Stage 3 — fact-conditioned rewrite
    t = time.monotonic()
    draft = generate_rewrite(facts, glossary, llm_fn)
    times["s3"] = int((time.monotonic() - t) * 1000)

    # Stage 4 — 3-tier verify-and-fix
    t = time.monotonic()
    final, report = verify_and_fix(draft, facts, source_text, llm_fn, glossary)
    times["s4"] = int((time.monotonic() - t) * 1000)

    return {
        "summary": final,
        "facts": facts,
        "glossary": [e.to_dict() for e in glossary],
        "sections_found": [
            {"category": s.category, "title": s.title}
            for s in preprocessed.sections
        ],
        "verification": {
            "missing_numbers":    report.missing_numbers,
            "wrong_laterality":   report.wrong_laterality,
            "wrong_organs":       report.wrong_organs,
            "wrong_drugs":        report.wrong_drugs,
            "missing_coverage":   report.missing_coverage,
            "unsupported_claims": report.unsupported_claims,
            "gray_zone_claims":   report.gray_zone_claims,
            "revised":            report.revised,
            "summary_str":        report.summary_str(),
        },
        "stage_times_ms": times,
    }


# ── Batch run ─────────────────────────────────────────────────────────────────

def run_batch(
    examples: list[dict],
    llm_fn: Callable[[str], str],
    num_examples: Optional[int] = None,
) -> list[dict]:
    """
    Run over a list of {"input": str, "target"/"gold": str} dicts.
    Shared UMLS resources are initialised once for the whole batch.
    """
    cache = APICache()

    umls_index: Optional[UMLSIndex] = None
    if config.UMLS_SUBSET_DB.exists():
        umls_index = UMLSIndex()

    uts_client: Optional[UTSClient] = None
    if config.UMLS_API_KEY:
        uts_client = UTSClient(cache=cache)

    if num_examples is not None:
        examples = examples[:num_examples]

    results = []
    for i, ex in enumerate(examples):
        print(f"  [{i+1}/{len(examples)}]", end="\r")
        try:
            out = run_pipeline(
                ex["input"], llm_fn,
                umls_index=umls_index,
                uts_client=uts_client,
                cache=cache,
            )
            results.append({
                "id": i,
                "input": ex["input"],
                "gold": ex.get("target", ex.get("gold", "")),
                "pred": out["summary"],
                "facts": out["facts"],
                "glossary_count": len(out["glossary"]),
                "verification": out["verification"],
                "stage_times_ms": out["stage_times_ms"],
            })
        except ValueError as exc:
            print(f"\n  [warn] example {i} failed stage 1: {exc}")
            results.append({
                "id": i,
                "input": ex["input"],
                "gold": ex.get("target", ex.get("gold", "")),
                "pred": "",
                "error": str(exc),
            })

        if (i + 1) % 50 == 0:
            print(f"\n  {i+1}/{len(examples)} done.")

    print()
    return results


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MedJarGone v4 pipeline")
    parser.add_argument("--model",  default=config.LLM_MODEL)
    parser.add_argument("--input",  required=True, help="Path to case report .txt")
    parser.add_argument("--output", default=None,  help="Save result JSON here")
    args = parser.parse_args()

    if not config.UMLS_API_KEY and not config.UMLS_SUBSET_DB.exists():
        print("[warn] No UMLS_API_KEY and no local UMLS subset — glossary will be empty.")
        print("  Set UMLS_API_KEY or run: python -m medjargone.setup.build_umls_index")

    source_text = Path(args.input).read_text(encoding="utf-8")
    llm_fn = load_llm(args.model)
    result = run_pipeline(source_text, llm_fn)

    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"Saved → {args.output}")
    else:
        print("\n=== Patient summary ===")
        print(result["summary"])
        print(f"\n=== Verification: {result['verification']['summary_str']} ===")
        print(f"Stage times (ms): {result['stage_times_ms']}")


if __name__ == "__main__":
    main()
