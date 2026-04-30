import inspect
import os
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel, PeftConfig
from peft import LoraConfig

# Monkey-patch LoraConfig to tolerate unknown kwargs from adapter configs saved
# by newer/custom peft versions (e.g. alora_invocation_tokens, qalora_group_size).
# PeftModel.from_pretrained also constructs LoraConfig internally, so a try/except
# around from_pretrained alone is not sufficient.
_lora_valid_keys = set(inspect.signature(LoraConfig.__init__).parameters.keys()) - {"self"}
_orig_lora_init = LoraConfig.__init__


def _tolerant_lora_init(self, **kwargs):
    _orig_lora_init(self, **{k: v for k, v in kwargs.items() if k in _lora_valid_keys})


LoraConfig.__init__ = _tolerant_lora_init


class Summarizer:
    def __init__(self, model_name):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        adapter_config_path = os.path.join(model_name, "adapter_config.json")
        dtype = torch.float16 if self.device == "cuda" else torch.float32

        if os.path.exists(adapter_config_path):
            peft_config = PeftConfig.from_pretrained(model_name)
            base_model_name = peft_config.base_model_name_or_path
            self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
            base_model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name, torch_dtype=dtype).to(self.device)
            peft_model = PeftModel.from_pretrained(base_model, model_name)
            # Merge LoRA weights into base model to eliminate per-layer adapter overhead
            self.model = peft_model.merge_and_unload().to(self.device)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=dtype).to(self.device)

        self.model.eval()

    def summarize(self, text, max_new_tokens=256):
        inputs = self.tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            summary_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                max_new_tokens=max_new_tokens,
                num_beams=4,
            )

        return self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
