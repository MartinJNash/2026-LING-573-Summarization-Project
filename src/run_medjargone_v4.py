"""
MedJarGone v4 batch inference — MultiClinSum test/train split.

Loads examples from the MultiClinSum dataset, runs the full v4 pipeline
(preprocess → fact extraction → UMLS glossary → rewrite → verify), and
saves results as a JSON file compatible with eval_pipeline.py.

Usage:
    python run_medjargone_v4.py
    python run_medjargone_v4.py --split train --num-examples 50
    python run_medjargone_v4.py --model Qwen/Qwen2.5-3B-Instruct \\
        --output results/outputs/medjargone-v4-test.json

Environment:
    export UMLS_API_KEY=...  # set before running
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from medjargone.pipeline.run import load_llm, run_batch
from medjargone import config


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
    parser.add_argument("--model",        default=config.LLM_MODEL)
    parser.add_argument("--output",       default=None,
                        help="Output JSON path (default: results/outputs/medjargone-v4-<split>.json)")
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

    print(f"Loading LLM: {args.model}")
    llm_fn = load_llm(args.model)

    print(f"Running v4 pipeline on {args.num_examples or len(examples)} examples...")
    results = run_batch(examples, llm_fn, num_examples=args.num_examples)

    output_path = Path(args.output) if args.output else (
        Path("results/outputs") / f"medjargone-v4-{args.split}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"model": args.model, "split": args.split, "examples": results}, f, indent=2)

    print(f"Saved {len(results)} results to {output_path}")


if __name__ == "__main__":
    main()
