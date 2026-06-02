"""
Download the MedJEx checkpoint and print its feature flags.

Usage:
    python scripts/inspect_medjex_checkpoint.py

If Binary_flag, TF_flag, MLM_flag are all False, no UMLS matcher is needed
and precompute_medjex_spans.py can run without MedCAT or QuickUMLS.
"""

import sys
import urllib.request
from pathlib import Path

CHECKPOINT_URL  = "https://huggingface.co/Mozzi/MedJEx/resolve/main/model.pth"
CHECKPOINT_PATH = Path(__file__).parents[1] / "src" / "medjargone" / "MedJEx" / "results" / "MedJEx.pth"

try:
    import torch
except ImportError:
    sys.exit("torch not found — run: pip install torch")


def main():
    if not CHECKPOINT_PATH.exists():
        print(f"Downloading checkpoint → {CHECKPOINT_PATH} (~500 MB)…", flush=True)
        CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(CHECKPOINT_URL, str(CHECKPOINT_PATH))
        print("Download complete.")
    else:
        print(f"Found existing checkpoint: {CHECKPOINT_PATH}")

    print("Loading checkpoint…", flush=True)
    ckpt = torch.load(str(CHECKPOINT_PATH), map_location="cpu", weights_only=False)

    print("\n=== MedJEx checkpoint flags ===")
    for key in ("Binary_flag", "TF_flag", "MLM_flag", "additional_feature"):
        print(f"  {key}: {ckpt.get(key, '<not found>')}")

    needs_umls = ckpt.get("Binary_flag") or ckpt.get("TF_flag") or ckpt.get("MLM_flag")
    print()
    if needs_umls:
        print("UMLS matcher REQUIRED — set up MedCAT or QuickUMLS before running")
        print("  precompute_medjex_spans.py --medcat-path <path>")
    else:
        print("No UMLS matcher needed — run precompute_medjex_spans.py without --medcat-path")
        print("  (pass --no-umls flag once precompute script is updated)")

    print("\nAll checkpoint keys:", [k for k in ckpt.keys() if k != "model_state_dict"])


if __name__ == "__main__":
    main()
