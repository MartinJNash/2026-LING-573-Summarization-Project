"""

Use an LLM on MultiClinSum examples to generate summaries, which are saved to JSON.

Usage:
    python llm_summarizer.py --model path/to/model/ --output results/outputs/output.json

"""

import vllm
import argparse
import json
import sys

sys.path.append("..")

from read_data import read_gs_training_data, read_test_training_data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to HuggingFace Transformers model to run inference with")
    parser.add_argument("--split", choices=["train", "test"], default="test", help="Which data split to run inference on (default: test)")
    parser.add_argument("--num-examples", type=int, default=None, help="Number of examples to run (default: all)")
    parser.add_argument("--output", default="outputs.json", help="Path to save inference outputs (default: outputs.json)")
    parser.add_argument("--percent_limit", type=int, default=20, help="Length of generated summary as a percentage of the source length (default: 20)")
    parser.add_argument("--max-model-len", type=int, default=8192, help="Max context length in tokens; reduce if OOM (default: 8192)")
    args = parser.parse_args()

    print(f"Loading data (split={args.split})...")
    loader = read_test_training_data if args.split == "test" else read_gs_training_data
    data = list(loader())

    if args.num_examples is not None:
        data = data[:args.num_examples]

    # read system prompt (plain instruction, no chat-format wrappers)
    print("Reading prompt template...")
    with open("./llm_only_prompt.txt", "r", encoding="utf-8") as prompt_file:
        system_prompt = prompt_file.read()

    if args.percent_limit <= 0 or args.percent_limit > 100:
        raise ValueError("The percent limit must be in the range [1, 100].")
    limit_statement = f"The length of the summary should be approximately {args.percent_limit}% that of the original length."
    system_prompt = system_prompt.replace("<<limit>>", limit_statement)

    # create an LLM
    # max_model_len caps context to avoid OOM on 11 GB 2080 Ti
    # (Qwen3.5's native context is 262k which would OOM immediately)
    print(f"Loading model: {args.model}...")
    llm = vllm.LLM(
        model=args.model,
        dtype="float16", # some GPUs do not meet min compute for Bfloat16
        gpu_memory_utilization=0.90,
        max_model_len=args.max_model_len,
    )

    # Recommended sampling params for Qwen3.5 non-thinking mode (text tasks)
    sampling_params = vllm.SamplingParams(
        temperature=1.0,
        top_p=1.0,
        top_k=20,
        presence_penalty=2.0,
        max_tokens=512,
    )

    # Build one conversation per example; llm.chat() applies the model's
    # chat template automatically (Qwen uses <|im_start|> format, not [INST])
    inputs = [ex["input"] for ex in data]
    conversations = [
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": inp},
        ]
        for inp in inputs
    ]

    # generate summaries
    print(f"Running inference on {len(data)} examples...\n")
    outputs = llm.chat(conversations, sampling_params)

    # build results
    results = []
    for i, (example, output) in enumerate(zip(data, outputs)):
        results.append({
            "id": i,
            "input": example["input"],
            "gold": example["target"],
            "pred": output.outputs[0].text,
        })

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(data)} done.")

    # write to output file
    with open(args.output, "w") as f:
        json.dump({
            "model": args.model,
            "split": args.split,
            "examples": results
        }, f, indent=2)

    print(f"\nSaved {len(results)} examples to {args.output}.")

if __name__ == "__main__":
    main()
