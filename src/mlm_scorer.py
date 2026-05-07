import torch
from transformers import BertForMaskedLM, BertTokenizer
from transformers import logging as hf_logging

class MLMScorer:
    """
    Masked Language Model scorer for computing MLM scores of spans using a BERT model.

    We know what tokens are present in the span
    We ask the model to predict the tokens in the span
    The model predicts logits for each token in vocabulary
    Because we know the correct token, we can inspect the LLM's logits for that correct token

    Softmax is between zero and 1. closer to zero is low likelihood of model choosing that token, 
    so more "surprise".

    Values between:
    - log(0) is undefined, but in practice is -inf 
    - log(1) == 0

    Negate that, and we get this: high MLM score means that LLM was less likely to predict that token,
    thus was 'surprised'
    """

    def __init__(self, model_name: str = "bert-base-uncased", device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        hf_logging.set_verbosity_error()  # Suppress transformers warnings

        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForMaskedLM.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def mlm_score(self, text: str, span_start: int, span_end: int) -> float | None:
        """
        Returns mean -log P for the tokens in text[span_start:span_end],
        conditioned on surrounding context. Returns None if the span
        tokenizes to zero BERT tokens.
        """

        # Convert text to tokens
        encoding = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            return_offsets_mapping=True,
        )

        # Position in original string
        offset_mapping = encoding["offset_mapping"][0].tolist()
        input_ids = encoding["input_ids"].to(self.device)

        # Find which BERT tokens overlap with the character span
        # get token indices that overlap with the inupt text span
        span_token_indices = [
            i
            for i, (tok_start, tok_end) in enumerate(offset_mapping)
            if tok_start < span_end and tok_end > span_start and tok_start < tok_end
        ]

        if not span_token_indices:
            return None

        # Make a copy of input ids (will use later)
        masked_ids = input_ids.clone()
        
        # Mask all span tokens at once and score them
        # Set all tokens in the span to [MASK] token id
        masked_ids[0, span_token_indices] = self.tokenizer.mask_token_id

        # Logits are the confidence score for each token in the vocab at each position
        with torch.no_grad():
            logits = self.model(
                input_ids=masked_ids,
                attention_mask=encoding["attention_mask"].to(self.device),
            ).logits

        # Compute log probabilities for the original tokens at the masked positions
        # Softmax changes all raw logits to probabilities from 0 to 1.
        log_probs = torch.log_softmax(logits[0], dim=-1)
        scores = [
            -log_probs[idx, input_ids[0, idx]].item()
            for idx in span_token_indices
        ]

        # Mean score across all tokens in the span
        return sum(scores) / len(scores)
