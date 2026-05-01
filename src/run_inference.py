"""
Compatibility entry point for `python -m src.run_inference`.
Preserves Martin's original CLI and JSONL output format (outputs.jsonl + meta.json).

For the newer JSON format (used by eval_pipeline.py), use run_inference.py directly.

Usage:
    python -m src.run_inference --lora-path models/bart-base-lora/best
    python -m src.run_inference --lora-path models/bart-base-lora/best --output-dir outputs/
"""

import argparse
import json
import torch
from dataclasses import dataclass
from pathlib import Path
from src.read_data import read_test_training_data
from src.model import Summarizer


@dataclass
class Config:
    lora_path: str
    max_examples: int | None
    output_dir: str
    batch_size: int


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora-path", required=True, help="Path to LoRA adapter directory (e.g. models/bart-base-lora/best)")
    parser.add_argument("--max-examples", type=int, default=None, help="Limit number of examples (default: all)")
    parser.add_argument("--output-dir", default="outputs", help="Directory to write outputs.jsonl and meta.json")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for inference (default: 8)")
    args = parser.parse_args()

    config = Config(
        lora_path=args.lora_path,
        max_examples=args.max_examples,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
    )
    run_inference(config)


def run_inference(config: Config):
    print(f"Loading model from {config.lora_path}...")
    summarizer = Summarizer(config.lora_path)

    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = list(read_test_training_data())
    if config.max_examples is not None:
        data = data[:config.max_examples]

    print(f"Running inference on {len(data)} examples (batch_size={config.batch_size})...\n")

    results = []
    for i in range(0, len(data), config.batch_size):
        batch = data[i:i + config.batch_size]
        inputs = summarizer.tokenizer(
            [ex["input"] for ex in batch],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        )
        inputs = {k: v.to(summarizer.device) for k, v in inputs.items()}

        with torch.no_grad():
            summary_ids = summarizer.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=256,
                num_beams=4,
            )

        preds = summarizer.tokenizer.batch_decode(summary_ids, skip_special_tokens=True)
        for j, (example, pred) in enumerate(zip(batch, preds)):
            results.append({"id": i + j, "gold": example["target"], "pred": pred})

        print(f"  {min(i + config.batch_size, len(data))}/{len(data)} done.")

    with open(out_dir / "outputs.jsonl", "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")

    with open(out_dir / "meta.json", "w") as f:
        json.dump({"model": config.lora_path, "num_examples": len(results)}, f, indent=2)

    print(f"\nSaved {len(results)} examples to {config.output_dir}/")


if __name__ == "__main__":
    main()
