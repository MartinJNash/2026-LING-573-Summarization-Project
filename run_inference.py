"""
Run inference on MultiClinSum examples and save outputs to JSON.
Run this once, then use eval_pipeline.py to compute metrics.

Usage:
    python run_inference.py --model path/to/model
    python run_inference.py --model path/to/model --split test --num-examples 100
    python run_inference.py --model path/to/model --output results/outputs.json
"""

import json
import argparse
from read_data import read_gs_training_data, read_test_training_data
from model import Summarizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path or HuggingFace name of the model to run inference with")
    parser.add_argument("--split", choices=["train", "test"], default="test", help="Which data split to run inference on (default: test)")
    parser.add_argument("--num-examples", type=int, default=None, help="Number of examples to run (default: all)")
    parser.add_argument("--output", default="outputs.json", help="Path to save inference outputs (default: outputs.json)")
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    summarizer = Summarizer(args.model)

    print(f"Loading data (split={args.split})...")
    loader = read_test_training_data if args.split == "test" else read_gs_training_data
    data = list(loader())
    if args.num_examples is not None:
        data = data[:args.num_examples]

    results = []
    print(f"Running inference on {len(data)} examples...\n")
    for i, example in enumerate(data):
        pred = summarizer.summarize(example["input"])
        results.append({
            "id": i,
            "input": example["input"],
            "gold": example["target"],
            "pred": pred,
        })
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(data)} done.")

    with open(args.output, "w") as f:
        json.dump({"model": args.model, "split": args.split, "examples": results}, f, indent=2)

    print(f"\nSaved {len(results)} examples to {args.output}")


if __name__ == "__main__":
    main()
