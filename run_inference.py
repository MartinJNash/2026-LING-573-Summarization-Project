"""
Run inference on MultiClinSum examples and save outputs to JSON.
Run this once, then use eval_pipeline.py to compute metrics.
"""

import json
from read_data import read_gs_training_data
from model import Summarizer

MODEL_NAME = "facebook/bart-base"
NUM_EXAMPLES = 5
OUTPUT_FILE = "outputs.json"


def main():
    print("Loading model...")
    summarizer = Summarizer(MODEL_NAME)

    print("Loading data...")
    data = list(read_gs_training_data())[:NUM_EXAMPLES]

    results = []
    print(f"Running inference on {NUM_EXAMPLES} examples...\n")
    for i, example in enumerate(data):
        pred = summarizer.summarize(example["input"])
        results.append({
            "id": i,
            "input": example["input"],
            "gold": example["target"],
            "pred": pred
        })
        print(f"Example {i+1} done.")

    with open(OUTPUT_FILE, "w") as f:
        json.dump({"model": MODEL_NAME, "examples": results}, f, indent=2)

    print(f"\nSaved {NUM_EXAMPLES} examples to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()