import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


class Summarizer:
    def __init__(self, model_name):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)

    def summarize(self, text, max_length=256):
        inputs = self.tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        summary_ids = self.model.generate(inputs["input_ids"], max_length=max_length)
        return self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
