"""
MedJarGone v4 batch inference — MultiClinSum test/train split.

Loads examples from the MultiClinSum dataset, runs the full v4 pipeline
(preprocess → fact extraction → UMLS glossary → rewrite → verify), and
saves results as a JSON file compatible with eval_pipeline.py.

Usage:
    python src/run_medjargone_v4.py
    python src/run_medjargone_v4.py --split train --num-examples 50
    python src/run_medjargone_v4.py --model Qwen/Qwen2.5-3B-Instruct \\
        --output results/v4/medjargone-v4-test.json

Environment:
    export UMLS_API_KEY=...  # set before running
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from medjargone_d4.pipeline.run import load_llm, run_batch
from medjargone_d4 import config


def _read_folder(root: Path):
    fulltext  = root / "fulltext"
    summaries = root / "summaries"
    for ft_path in sorted(fulltext.iterdir()):
        if ft_path.is_file():
            sum_path = summaries / (ft_path.stem + "_sum.txt")
            yield {
                "input":  ft_path.read_text(encoding="utf-8"),
                "target": sum_path.read_text(encoding="utf-8") if sum_path.exists() else "",
            }


def main():
    parser = argparse.ArgumentParser(description="MedJarGone v4 batch inference")
    parser.add_argument("--split",        choices=["train", "test"], default="test")
    parser.add_argument("--num-examples", type=int, default=None)
    parser.add_argument("--start-idx",    type=int, default=0,
                        help="First example index (for job-array slicing)")
    parser.add_argument("--model",         default=config.LLM_MODEL)
    parser.add_argument("--output",        default=None,
                        help="Output JSON path (default: results/v4/medjargone-v4-<split>.json)")
    parser.add_argument("--medjex-spans",  default=str(config.MEDJEX_SPANS_TEST),
                        help="Precomputed MedJEx spans JSON (default: data/medjargone/medjex_spans_test.json)")
    args = parser.parse_args()

    if not config.UMLS_API_KEY and not config.UMLS_SUBSET_DB.exists():
        print("[warn] No UMLS_API_KEY and no local UMLS subset — glossary will be empty.")
        print("  Set: export UMLS_API_KEY=<key>")

    data_dir = (config.MULTICLINSUM_TEST_DIR
                if args.split == "test"
                else config.MULTICLINSUM_TRAIN_DIR)
    if not data_dir.exists():
        print(f"[error] Dataset not found at {data_dir}")
        sys.exit(1)

    examples = list(_read_folder(data_dir))
    print(f"Loaded {len(examples)} examples from {data_dir}")

    start = args.start_idx
    end   = start + args.num_examples if args.num_examples is not None else len(examples)
    slice_examples = examples[start:end]
    print(f"Running v4 pipeline on examples {start}–{min(end, len(examples))-1} "
          f"({len(slice_examples)} total)...")

    medjex_spans = None
    spans_path = Path(args.medjex_spans)
    if spans_path.exists():
        with open(spans_path, encoding="utf-8") as f:
            medjex_spans = json.load(f)
        print(f"Loaded MedJEx spans for {len(medjex_spans)} docs from {spans_path}")
    else:
        print(f"[warn] MedJEx spans not found at {spans_path} — running without jargon seeding")

    print(f"Loading LLM: {args.model}")
    llm_fn = load_llm(args.model)

    results = run_batch(slice_examples, llm_fn, medjex_spans=medjex_spans, start_idx=start)
    # Rewrite local IDs to absolute dataset indices
    for r in results:
        r["id"] += start

    output_path = Path(args.output) if args.output else (
        Path("results/v4") / f"medjargone-v4-{args.split}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"model": args.model, "split": args.split, "examples": results}, f, indent=2)

    print(f"Saved {len(results)} results to {output_path}")


if __name__ == "__main__":
    main()
