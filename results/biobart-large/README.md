---
base_model: GanjinZero/biobart-v2-large
library_name: peft
tags:
- base_model:adapter:GanjinZero/biobart-v2-large
- lora
- transformers
- summarization
- medical
language:
- en
---

# BioBART-v2-large + LoRA — MultiClinSum

LoRA adapter fine-tuned on the [MultiClinSum](https://zenodo.org/records/17341582) GS training split for abstractive summarization of clinical notes into plain-language patient summaries. Part of UW LING 573 (Spring 2026) — MedJarGone project.

## Model Details

- **Base model:** `GanjinZero/biobart-v2-large` (~406M parameters)
- **Fine-tuning method:** LoRA (PEFT) — only adapter weights are stored here
- **Trainable parameters:** ~2.36M / 406M total (~0.58%)
- **LoRA config:** r=16, α=16, dropout=0.1, targets: q_proj + v_proj
- **Task:** Abstractive summarization (seq2seq)
- **Language:** English

## Training

- **Dataset:** MultiClinSum GS split — 594 examples (476 train / 53 val, 90/10 split, seed=42)
- **Epochs:** 10
- **Batch size:** 4 (per device)
- **Learning rate:** 5e-5, weight decay 0.01
- **Precision:** fp16
- **Checkpoint selection:** best BERTScore F1 on validation set
- **Hardware:** Hyak GPU cluster (UW), ~31 minutes on a single NVIDIA GPU
- **Framework:** PEFT 0.19.1, Transformers

## Usage

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

base = "GanjinZero/biobart-v2-large"
adapter = "Pika4028/biobart-v2-large-multiclinsum-lora"

tokenizer = AutoTokenizer.from_pretrained(base)
model = AutoModelForSeq2SeqLM.from_pretrained(base, torch_dtype="auto")
model = PeftModel.from_pretrained(model, adapter)
model = model.merge_and_unload()  # merge LoRA weights for inference

inputs = tokenizer("Patient clinical note here...", return_tensors="pt", truncation=True, max_length=1024)
summary_ids = model.generate(**inputs, max_new_tokens=256, num_beams=4)
print(tokenizer.decode(summary_ids[0], skip_special_tokens=True))
```

## Evaluation

Evaluated on the full MultiClinSum test set (3,396 examples). Inference output at `results/outputs/biobart-large-lora.json`. See [MODELS.md](../../MODELS.md) for results.

Primary metric: rougeLsum (matches MultiClinSum shared task scoring).
