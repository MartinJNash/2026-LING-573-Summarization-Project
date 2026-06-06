"""
Run all test levels in order. Stop-on-first-failure is off — each level
runs regardless so you can see exactly which components are working.

Usage:
    cd 2026-LING-573-Summarization-Project
    python -m medjargone.tests.run_all
    python -m medjargone.tests.run_all --level 0   # run only level 0
    python -m medjargone.tests.run_all --level 3   # run only UMLS API test
"""

import argparse
import importlib
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Suppress PyRuSH sentence-splitter debug noise from medspaCy
logging.getLogger("PyRuSH").setLevel(logging.WARNING)

LEVELS = [
    (0, "medjargone.tests.test_00_config",          "Config & paths"),
    (1, "medjargone.tests.test_01_preprocess",      "medspaCy preprocessing"),
    (2, "medjargone.tests.test_02_verify_rules",    "Tier-1 rule checks"),
    (3, "medjargone.tests.test_03_umls_api",        "UMLS API (needs key)"),
    (4, "medjargone.tests.test_04_glossary",        "Glossary subsystem (needs UMLS)"),
    (5, "medjargone.tests.test_05_pipeline_mock",   "Full pipeline — mock LLM"),
]


def run_level(mod_name: str, label: str):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    try:
        mod = importlib.import_module(mod_name)
        # Call every function whose name starts with 'test_'
        fns = [v for k, v in sorted(vars(mod).items())
               if k.startswith("test_") and callable(v)]
        for fn in fns:
            try:
                fn()
            except SystemExit:
                pass
            except Exception as e:
                print(f"  ERR  {fn.__name__} — {e}")
    except ImportError as e:
        print(f"  ERR  Could not import {mod_name}: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", type=int, default=None,
                        help="Run only this level (0–5). Omit to run all.")
    args = parser.parse_args()

    to_run = [(n, m, l) for n, m, l in LEVELS
              if args.level is None or n == args.level]

    if not to_run:
        print(f"Unknown level {args.level}. Choose 0–5.")
        sys.exit(1)

    for n, mod_name, label in to_run:
        run_level(mod_name, f"Level {n}: {label}")

    print(f"\n{'='*60}")
    print("  Done.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
