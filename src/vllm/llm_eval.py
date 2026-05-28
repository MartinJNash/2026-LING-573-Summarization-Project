"""
LLM-based evaluation of MedJarGone summarization outputs using NVIDIA NIM.
Scores each (source, generated) pair on informativeness, simplification, faithfulness.

Usage:
    python llm_eval.py --input results/outputs/biobart-large-lora.json
    python llm_eval.py --input results/outputs/biobart-large-lora.json --output llm_eval_results.csv --max 50
"""

import os
import json
import time
import logging
import argparse
import pandas as pd
from openai import OpenAI, RateLimitError

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

client = OpenAI(
    api_key=os.environ.get("NVIDIA_API_KEY"),
    base_url="https://integrate.api.nvidia.com/v1",
)

MODEL = "meta/llama-3.1-70b-instruct"
RATE_LIMIT_RPM = 20  # conservative — avoids 429s from burst traffic
SLEEP_BETWEEN = 60.0 / RATE_LIMIT_RPM  # 3s

PROMPT_TEMPLATE = """Imagine you are a human annotator. Evaluate the quality of a generated plain language summary of a clinical document on the following criteria:

1. Informativeness: does it cover key findings, diagnosis, and treatment plan without hallucinating?
2. Simplification: simple vocabulary, minimal jargon, accessible to a lay reader?
3. Faithfulness: factually aligned with source, no substitutions or errors?

Return ONLY a JSON object with keys: informativeness, simplification, faithfulness. Scores range from 0 (worst) to 100 (best). No explanation, no preamble.

Clinical source document: {source}
Generated plain language summary: {generated}"""


def score_summary(source: str, generated: str, max_retries: int = 5) -> dict | None:
    prompt = PROMPT_TEMPLATE.format(source=source, generated=generated)
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=64,
            )
            raw = response.choices[0].message.content.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            scores = json.loads(raw)
            return {
                "informativeness": float(scores["informativeness"]),
                "simplification":  float(scores["simplification"]),
                "faithfulness":    float(scores["faithfulness"]),
            }
        except RateLimitError:
            wait = 60 * (attempt + 1)
            logger.warning(f"Rate limited — waiting {wait}s before retry {attempt + 1}/{max_retries}")
            time.sleep(wait)
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse failed: {e} | raw response: {raw!r}")
            return None
        except Exception as e:
            logger.error(f"API call failed: {e}")
            return None
    logger.error("Max retries exceeded, skipping example.")
    return None


def run_eval_on_dataset(df: pd.DataFrame, output_path: str) -> pd.DataFrame:
    """
    Expects columns: source, generated.
    Writes results incrementally to output_path after each example.
    Returns df with added columns: llm_informativeness, llm_simplification, llm_faithfulness.
    """
    results = []
    n = len(df)
    write_header = True

    for i, row in df.iterrows():
        logger.info(f"Scoring {i + 1}/{n}...")
        scores = score_summary(row["source"], row["generated"])
        result = scores if scores else {"informativeness": None, "simplification": None, "faithfulness": None}
        results.append(result)

        # Write this row immediately to CSV
        row_out = row.to_dict()
        row_out["llm_informativeness"] = result["informativeness"]
        row_out["llm_simplification"]  = result["simplification"]
        row_out["llm_faithfulness"]    = result["faithfulness"]
        pd.DataFrame([row_out]).to_csv(output_path, mode="a", header=write_header, index=False)
        write_header = False

        if i < n - 1:
            time.sleep(SLEEP_BETWEEN)

    scores_df = pd.DataFrame(results)
    df = df.copy()
    df["llm_informativeness"] = scores_df["informativeness"].values
    df["llm_simplification"]  = scores_df["simplification"].values
    df["llm_faithfulness"]    = scores_df["faithfulness"].values
    return df


def load_examples(path: str, max_examples: int | None = None) -> pd.DataFrame:
    with open(path) as f:
        content = f.read()
    try:
        data = json.loads(content)
        examples = data["examples"]
    except (json.JSONDecodeError, KeyError):
        examples = [json.loads(line) for line in content.splitlines() if line.strip()]

    if max_examples:
        examples = examples[:max_examples]

    rows = []
    for ex in examples:
        if "input" not in ex:
            raise ValueError("Examples must have an 'input' field for LLM eval (source text required).")
        rows.append({"source": ex["input"], "generated": ex["pred"], "gold": ex["gold"]})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to inference outputs JSON (must have 'input' field)")
    parser.add_argument("--output", default="llm_eval_results.csv", help="Path to save results CSV")
    parser.add_argument("--max", type=int, default=None, help="Max examples to evaluate (default: all)")
    args = parser.parse_args()

    if not os.environ.get("NVIDIA_API_KEY"):
        raise EnvironmentError("NVIDIA_API_KEY not set. Add it to your .env and source it.")

    logger.info(f"Loading examples from {args.input}...")
    df = load_examples(args.input, max_examples=args.max)
    logger.info(f"Evaluating {len(df)} examples at {RATE_LIMIT_RPM} RPM (~{len(df) * SLEEP_BETWEEN / 60:.1f} min)...")

    df = run_eval_on_dataset(df, output_path=args.output)

    logger.info(f"Saved to {args.output}")

    print(f"\n========== LLM EVAL RESULTS ==========")
    print(f"Examples: {len(df)}")
    for col in ["llm_informativeness", "llm_simplification", "llm_faithfulness"]:
        print(f"  {col}: {df[col].mean():.2f} (mean)")


if __name__ == "__main__":
    main()


# --- Example: run on our test set outputs ---
#
# Load NVIDIA_API_KEY first:
#   export NVIDIA_API_KEY=$(grep NVIDIA_API_KEY .env | cut -d= -f2)
#
# Run on biobart-large-lora full outputs (has 'input' field), first 50 examples:
#   python llm_eval.py --input results/outputs/biobart-large-lora.json --max 50
#
# Run on all 3396 examples (~85 min at 40 RPM):
#   python llm_eval.py --input results/outputs/biobart-large-lora.json --output llm_eval_biobart_large.csv
