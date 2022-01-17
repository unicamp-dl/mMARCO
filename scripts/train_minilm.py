import torch
import argparse
from datasets import load_dataset
from transformers import (
    Trainer,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoConfig,
    TrainingArguments,
)

LANGUAGES = [
    "arabic",
    "chinese",
    "dutch",
    "english",
    "french",
    "german",
    "hindi",
    "indonesian",
    "italian",
    "japanese",
    "portuguese",
    "russian",
    "spanish",
    "vietnamese",
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--model", default="nreimers/mMiniLMv2-L6-H384-distilled-from-XLMR-Large", type=str, required=False
    )
    parser.add_argument("--language", default="english", choices=LANGUAGES)
    parser.add_argument("--learning_rate", default=2e-5, type=float, required=False)
    parser.add_argument("--batch_size", default=64, type=int, required=False)
    parser.add_argument("--max_samples", default=None, type=int, required=False)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    config = AutoConfig.from_pretrained(args.model)
    config.num_labels = 1
    config.problem_type = "multi_label_classification"
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, config=config
    )

    dataset = load_dataset("unicamp-dl/mmarco", args.language)["train"]
    if args.max_samples:
        dataset = dataset.select(range(args.max_samples))

    def split_examples(batch):
        queries = []
        passages = []
        labels = []
        for label in ["positive", "negative"]:
            for (query, passage) in zip(batch["query"], batch[label]):
                queries.append(query)
                passages.append(passage)
                labels.append(int(label == "positive"))
        return {"query": queries, "passage": passages, "label": labels}

    dataset = dataset.map(
        split_examples, batched=True, remove_columns=["positive", "negative"]
    )

    def tokenize(batch):
        tokenized = tokenizer(
            batch["query"],
            batch["passage"],
            padding=True,
            truncation="only_second",
            max_length=512,
        )
        tokenized["labels"] = [[float(label)] for label in batch["label"]]
        return tokenized

    dataset = dataset.map(
        tokenize, batched=True, remove_columns=["query", "passage", "label"]
    )
    dataset.set_format("torch")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        fp16=True,
        fp16_backend="amp",
        per_device_train_batch_size=args.batch_size,
        logging_steps=500,
        warmup_steps=100,
        save_total_limit=1,
        num_train_epochs=1,
    )
    trainer = Trainer(
        model,
        training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )

    train_result = trainer.train()
    trainer.save_model()
