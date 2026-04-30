import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--use-peft", action="store_true", default=False)
args = parser.parse_args()

print(f"use_peft: {args.use_peft}")
