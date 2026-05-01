"""
Compatibility entry point for `python -m src.train`.
Accepts Martin's original CLI interface and delegates to pipeline.train().

Args are identical to pipeline.py except --num-epochs defaults to 3 here
(matching Martin's original default) vs 10 in pipeline.py (Priyanshi's default).
"""

import argparse
from pipeline import Config, train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="facebook/bart-base", help="Path to base model")
    parser.add_argument("--use-peft", action="store_true", default=False)
    parser.add_argument("--output-dir", default="./results/final_model", help="Path to output directory")
    parser.add_argument("--num-epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit dataset size for smoke testing")
    parser.add_argument("--dataset", default="gs", choices=["gs", "large-scale"], help="Training split: gs (594 examples) or large-scale")
    parser.add_argument("--batch-size", type=int, default=4, help="Per-device train batch size")
    args = parser.parse_args()

    config = Config(
        base_model=args.base_model,
        use_peft=args.use_peft,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        max_samples=args.max_samples,
        dataset=args.dataset,
    )
    train(config)


if __name__ == "__main__":
    main()
