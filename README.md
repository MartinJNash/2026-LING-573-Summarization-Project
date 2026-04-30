MedJarGone - Summarize medical notes and remove technical terms.


# running

```sh
pyton -m src.train
  --base-model hf-model-name
  --output-dir path/to/results/dir
  --num-epochs 3
  --use-peft  
```

```sh
python -m src.run_inference
  --lora-path path/to/models 
  --max-examples 300 
  --output-dir path/to/results/dir
```

```sh
python -m src.eval_pipeline 
  --input path/to/outputs.jsonl
  --output path/to/results/dir
```
