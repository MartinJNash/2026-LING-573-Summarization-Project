from read_data import read_gs_training_data
from datasets import Dataset
from transformers import Seq2SeqTrainingArguments
from transformers import Seq2SeqTrainer
from transformers import DataCollatorForSeq2Seq
from model import Summarizer
import evaluate
import numpy as np


MODEL_NAME = "facebook/bart-base"

def main():
    summarizer = Summarizer(MODEL_NAME)
    tokenizer = summarizer.tokenizer
    model = summarizer.model

    ds = Dataset.from_generator(read_gs_training_data)
    split = ds.train_test_split(test_size=0.1, seed=42)

    def preprocess(examples):
        inputs = tokenizer(examples["input"], max_length=1024, truncation=True)
        targets = tokenizer(text_target=examples["target"], max_length=256, truncation=True)
        inputs["labels"] = targets["input_ids"]
        return inputs

    tokenized = split.map(preprocess, batched=True, remove_columns=["input", "target"])
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    training_args = Seq2SeqTrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=4,
        num_train_epochs=10,
        weight_decay=0.01,
        predict_with_generate=True,
        generation_max_length=256,
        load_best_model_at_end=True,
        metric_for_best_model="rouge1",
        save_total_limit=2,
        logging_steps=50,
        report_to="none",
        seed=42,
        dataloader_pin_memory=False,
        dataloader_num_workers=2,
    )

    rouge = evaluate.load("rouge")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.clip(predictions, 0, tokenizer.vocab_size - 1)
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [label.strip() for label in decoded_labels]
        result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        return {k: round(v * 100, 4) for k, v in result.items()}

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model("./results/final_model")


if __name__ == "__main__":
    main()
    print("done")
