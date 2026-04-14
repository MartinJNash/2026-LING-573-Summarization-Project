import transformers
import torch

print(transformers.__version__)
print(torch.__version__)

from transformers import BartForConditionalGeneration, BartTokenizer

model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")


ARTICLE = """
The Amazon rainforest is a massive, broad-leafed tropical forest in the Amazon biome
that covers most of the Amazon basin of South America. It is the world's largest
tropical rainforest, famed for its immense biodiversity. It is crisscrossed by
thousands of rivers, including the powerful Amazon. The rainforest is crucial for
climate regulation and is home to thousands of unique species of flora and fauna.
However, deforestation poses a significant threat to its survival, driven by
agriculture and logging activities.
"""


inputs = tokenizer(f"Summarize: {ARTICLE}", return_tensors="pt")
summary_ids = model.generate(
    inputs["input_ids"],
    forced_bos_token_id=0,
    # min_new_tokens=200,
    # max_new_tokens=300,
)
print(tokenizer.decode(summary_ids[0], skip_special_tokens=True))
