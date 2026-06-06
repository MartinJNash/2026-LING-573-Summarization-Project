"""
Merge per-chunk outputs from run_medjargone_v4_array_hyak.sh into a single file.

Usage:
    python scripts/merge_v4_chunks.py
    python scripts/merge_v4_chunks.py --chunks-dir results/outputs/chunks \
        --output results/outputs/medjargone-v4-test.json
"""

import argparse
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks-dir", default="results/outputs/chunks")
    parser.add_argument("--output",     default="results/outputs/medjargone-v4-test.json")
    parser.add_argument("--num-chunks", type=int, default=16)
    args = parser.parse_args()

    chunks_dir = Path(args.chunks_dir)
    all_examples = []
    model = None
    missing = []

    for i in range(args.num_chunks):
        path = chunks_dir / f"medjargone-v4-test-chunk-{i}.json"
        if not path.exists():
            missing.append(i)
            print(f"[warn] chunk {i} missing: {path}")
            continue
        data = json.loads(path.read_text())
        if model is None:
            model = data.get("model")
        all_examples.extend(data["examples"])

    if missing:
        print(f"[warn] {len(missing)} chunks missing — output will be incomplete.")

    all_examples.sort(key=lambda x: x["id"])
    print(f"Merged {len(all_examples)} examples from {args.num_chunks - len(missing)} chunks.")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"model": model, "split": "test", "examples": all_examples}, indent=2))
    print(f"Saved → {out}")

    if missing:
        sys.exit(1)


if __name__ == "__main__":
    main()
