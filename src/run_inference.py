"""
Run inference on MultiClinSum examples and save outputs to JSONL.
Run this once, then use eval_pipeline.py to compute metrics.
"""

import argparse
import json
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from src.read_data import read_test_training_data
from src.model import Summarizer
from peft import PeftModel, PeftConfig


@dataclass
class Config:
    lora_path: str
    max_examples: int | None
    output_dir: str


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora-path", required=True, help="Path to LoRA adapter directory (e.g. models/bart-base-lora/best)")
    parser.add_argument("--max-examples", type=int, default=None, help="Limit number of examples (default: all)")
    parser.add_argument("--output-dir", default="outputs", help="Directory to write outputs.jsonl and meta.json")
    args = parser.parse_args()

    config = Config(
        lora_path=args.lora_path,
        max_examples=args.max_examples,
        output_dir=args.output_dir,
    )
    run_inference(config)


def run_inference(config: Config):
    peft_config = PeftConfig.from_pretrained(config.lora_path)
    base_model_name = peft_config.base_model_name_or_path

    print(f"Loading base model {base_model_name}...")
    summarizer = Summarizer(base_model_name)

    print(f"Applying LoRA adapter from {config.lora_path}...")
    summarizer.model = PeftModel.from_pretrained(summarizer.model, config.lora_path)
    summarizer.model = summarizer.model.merge_and_unload()

    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = read_test_training_data()
    if config.max_examples is not None:
        data = islice(data, config.max_examples)

    print("Running inference...\n")
    count = 0
    with open(out_dir / "outputs.jsonl", "w") as f:
        for i, example in enumerate(data):
            pred = summarizer.summarize(example["input"])
            f.write(json.dumps({"id": i, "gold": example["target"], "pred": pred}) + "\n")
            print(f"Example {i+1} done.")
            count += 1

    with open(out_dir / "meta.json", "w") as f:
        json.dump({"model": config.lora_path, "base_model": base_model_name, "num_examples": count}, f, indent=2)

    print(f"\nSaved {count} examples to {config.output_dir}/")


if __name__ == "__main__":
    main()