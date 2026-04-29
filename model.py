import inspect
import json
import os
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel, PeftConfig
from peft import LoraConfig


def _load_peft_config_tolerant(model_name):
    """Load PeftConfig, ignoring unknown fields that newer peft versions may have added."""
    adapter_config_path = os.path.join(model_name, "adapter_config.json")
    with open(adapter_config_path) as f:
        config_dict = json.load(f)

    try:
        return PeftConfig.from_pretrained(model_name)
    except TypeError:
        # Strip keys not accepted by LoraConfig.__init__ and retry
        valid_keys = set(inspect.signature(LoraConfig.__init__).parameters.keys()) - {"self"}
        stripped = {k: v for k, v in config_dict.items() if k in valid_keys}
        return LoraConfig(**stripped)


class Summarizer:
    def __init__(self, model_name):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Detect PEFT adapter by presence of adapter_config.json
        adapter_config_path = os.path.join(model_name, "adapter_config.json")

        dtype = torch.float16 if self.device == "cuda" else torch.float32

        if os.path.exists(adapter_config_path):
            peft_config = _load_peft_config_tolerant(model_name)
            base_model_name = peft_config.base_model_name_or_path

            self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
            base_model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name, dtype=dtype).to(self.device)
            peft_model = PeftModel.from_pretrained(base_model, model_name)
            # Merge LoRA weights into base model to eliminate per-layer adapter overhead
            self.model = peft_model.merge_and_unload().to(self.device)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, dtype=dtype).to(self.device)

        self.model.eval()

    def summarize(self, text, max_new_tokens=256):
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=1024,
            truncation=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            summary_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                max_new_tokens=max_new_tokens,
                num_beams=4,
            )

        return self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    def summarize_batch(self, texts, max_new_tokens=256, batch_size=8):
        outputs = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1024,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                summary_ids = self.model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                    max_new_tokens=max_new_tokens,
                    num_beams=4,
                )

            outputs.extend(
                self.tokenizer.batch_decode(summary_ids, skip_special_tokens=True)
            )

        return outputs
