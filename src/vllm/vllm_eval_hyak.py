"""
LLM-based evaluation of MedJarGone summarization outputs using vLLM on UW RCC's Hyak.
Scores each (source, generated) pair on informativeness, simplification, and faithfulness.

Usage:
    python llm_eval.py --input results/outputs/biobart-large-lora.json
    python llm_eval.py --input results/outputs/biobart-large-lora.json --output llm_eval_results.csv --max 50
"""

import json
import logging
import argparse
import numpy as np
import pandas as pd
import vllm

USER_PROMPT_TEMPLATE = """Clinical source document: {source}
Generated plain language summary: {generated}"""

MAX_MODEL_LEN = 8192 # using default from our previous vLLM Python scripts

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

def run_eval_on_dataset(model, df: pd.DataFrame, output_path: str):
    """
    Expects df columns: source, generated.
    Writes scores to a JSON file with the associated example ID.
    Returns results, which is a list of score dictionaries.
    """

    # create an LLM
    print(f"Loading model: {model}...")
    llm = vllm.LLM(
        model=model,
        dtype="float16", # some GPUs do not meet min compute for Bfloat16
        gpu_memory_utilization=0.90,
        max_model_len=MAX_MODEL_LEN,
    )

    # Recommended sampling params for Qwen3.5 non-thinking mode (text tasks)
    sampling_params = vllm.SamplingParams(
        temperature=1.0,
        top_p=1.0,
        top_k=20,
        presence_penalty=2.0,
        max_tokens=512,
    )
    
    logger.info("Reading prompt template...")
    with open("../prompts/llm_eval_prompt.txt", "r", encoding="utf-8") as prompt_file:
        SYSTEM_PROMPT_TEMPLATE = prompt_file.read()

    # Build one conversation per example; llm.chat() applies the model's

    conversations = [
        [
            {"role": "system", "content": SYSTEM_PROMPT_TEMPLATE},
            {"role": "user", "content": USER_PROMPT_TEMPLATE.format(source=source, generated=generated)}
        ]
        for source, generated in zip(df["source"], df["generated"])
    ]

    # generate scores
    print(f"Generating scores on {len(conversations)} examples...")
    outputs = llm.chat(conversations, sampling_params)

    raw_scores = [output.outputs[0].text.strip() for output in outputs]

    # preprocess outputs if necessary
    scores_dicts = []
    for rs in raw_scores:
        try:
            if rs.startswith("```"):
                rs = rs.split("```")[1]
                if rs.startswith("json"):
                    rs = rs[4:]
            scores = json.loads(rs)
            scores_dict = {
                "informativeness": float(scores["informativeness"]),
                "simplification":  float(scores["simplification"]),
                "faithfulness":    float(scores["faithfulness"]),
            }
            scores_dicts.append(scores_dict)
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse failed: {e} | raw response: {rs!r}")
            return None
    

    # build results
    results = []
    for i, scores_dict in enumerate(scores_dicts):
        results.append({
            "id": i,
            "informativeness": scores_dict["informativeness"],
            "simplification": scores_dict["simplification"],
            "faithfulness": scores_dict["faithfulness"],
        })
    
    # write to final outputs JSON
    with open(output_path, "w") as f:
        json.dump({
            "model": model,
            "examples": results
        }, f, indent=2)
    
    return results


def load_examples(path: str, num_examples: int | None = None) -> pd.DataFrame:
    with open(path) as f:
        content = f.read()
    try:
        data = json.loads(content)
        examples = data["examples"]
    except (json.JSONDecodeError, KeyError):
        examples = [json.loads(line) for line in content.splitlines() if line.strip()]

    if num_examples:
        examples = examples[:num_examples]

    rows = []
    for ex in examples:
        if "input" not in ex:
            raise ValueError("Examples must have an 'input' field for LLM eval (source text required).")
        rows.append({"source": ex["input"], "generated": ex["pred"]})
    examples_df = pd.DataFrame(rows)
    return examples_df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to HuggingFace Transformers model to run inference with")
    parser.add_argument("--input", required=True, help="Path to inference outputs JSON (must have 'input' field!)")
    parser.add_argument("--output", default="vllm_eval_results.json", help="Path to save results JSON (default: llm_eval_results.json)")
    parser.add_argument("--num_examples", type=int, default=None, help="Number of examples to evaluate (default: all)")
    args = parser.parse_args()

    logger.info(f"Loading examples from {args.input}...")
    df = load_examples(args.input, num_examples=args.num_examples)

    results = run_eval_on_dataset(args.model, df, args.output)

    logger.info(f"Saved to {args.output}.")

    llm_inform = np.mean([ex["informativeness"] for ex in results])
    llm_simp = np.mean([ex["simplification"] for ex in results])
    llm_faith = np.mean([ex["faithfulness"] for ex in results])

    print(f"\n========== LLM EVAL RESULTS ==========")
    print(f"Examples: {len(df)}")
    print(f"  Informativeness: {llm_inform:.2f} (mean)")
    print(f"  Simplification: {llm_simp:.2f} (mean)")
    print(f"  Faithfulness: {llm_faith:.2f} (mean)")


if __name__ == "__main__":
    main()