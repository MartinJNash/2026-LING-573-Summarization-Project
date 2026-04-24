import argparse
from read_data import read_gs_training_data
from datasets import Dataset
from transformers import Seq2SeqTrainingArguments
from transformers import Seq2SeqTrainer
from transformers import DataCollatorForSeq2Seq
from model import Summarizer
import evaluate
import numpy as np
from bert_score import score as bert_score_fn
from peft import LoraConfig, get_peft_model
from dataclasses import dataclass

@dataclass
class Config:
    base_model: str
    use_peft: bool
    output_dir: str
    batch_size: int = 4

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="facebook/bart-base", help="Path to base model")
    parser.add_argument("--use-peft", action="store_true", default=False)
    parser.add_argument("--output-dir", default="./results/final_model", help="Path to output directory")
    parser.add_argument("--batch-size", type=int, default=4, help="Per-device train batch size")
    args = parser.parse_args()

    config = Config(
        base_model=args.base_model,
        use_peft=args.use_peft,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
    )
    train(config)

def train(config: Config):
    summarizer = Summarizer(config.base_model)
    tokenizer = summarizer.tokenizer
    model = summarizer.model

    # PEFT with LoRA
    if config.use_peft:
        lora_config = LoraConfig(
            r=16,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.1,
            bias="none",
            modules_to_save=["classifier"],
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

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
        output_dir=config.output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=config.batch_size,
        num_train_epochs=10,
        weight_decay=0.01,
        fp16=True,
        predict_with_generate=True,
        generation_max_length=256,
        load_best_model_at_end=True,
        metric_for_best_model="bertscore_f1",
        greater_is_better=True,
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

        rouge_result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        metrics = {k: round(v * 100, 4) for k, v in rouge_result.items()}

        # BERTScore — used for checkpoint selection (captures semantic similarity
        # that ROUGE misses; Angulo & Yeste found ROUGE/BERTScore negatively correlated r=-0.49)
        _, _, F1 = bert_score_fn(
            decoded_preds, decoded_labels,
            lang="en",
            model_type="distilbert-base-uncased",
            verbose=False,
        )
        metrics["bertscore_f1"] = round(F1.mean().item(), 4)

        return metrics

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
    trainer.save_model(config.output_dir)


if __name__ == "__main__":
    main()
    print("done")
