"""
Fine-tunes a pretrained transformer (DistilBERT by default; Legal-BERT as an
option) on the same labeled clauses the TF-IDF baseline used, with the SAME
train/test split logic (random_state=42) so results are directly comparable.

This is the second leg of the three-way comparison:
  1. TF-IDF + Logistic Regression  (src/train_classifier.py)
  2. Fine-tuned transformer        (this script)
  3. Zero-shot LLM classification  (src/zero_shot_llm.py, next)

Usage:
    python src/train_transformer.py                  # DistilBERT (default, faster)
    python src/train_transformer.py --model legalbert # Legal-BERT (slower, legal-domain pretrained)

Honest note for your report: fine-tuning on ~250 examples across 5 classes
is small for a transformer — some categories (forced_arbitration: 6 examples,
penalty_obscurity: 11) may not train reliably. That instability is itself a
valid, citable finding (mirrors the exact question the "Text to Trust" paper
asks about fine-tuning vs. zero-shot under data scarcity), not a bug to hide.
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)

from categories import CATEGORY_IDS

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "labeled" / "clauses_reviewed.csv"
MODEL_OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "labeled" / "transformer_model"

MODEL_CHOICES = {
    "distilbert": "distilbert-base-uncased",
    "legalbert": "nlpaueb/legal-bert-base-uncased",
}

LABEL2ID = {label: i for i, label in enumerate(CATEGORY_IDS)}
ID2LABEL = {i: label for label, i in LABEL2ID.items()}


class ClauseDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


def load_data():
    texts, labels = [], []
    with open(DATA_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["label"]:
                texts.append(row["clause_text"])
                labels.append(LABEL2ID[row["label"]])
    return texts, labels


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    report = classification_report(
        labels, preds, labels=list(ID2LABEL.keys()),
        target_names=list(ID2LABEL.values()), output_dict=True, zero_division=0,
    )
    return {"accuracy": report["accuracy"], "macro_f1": report["macro avg"]["f1-score"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODEL_CHOICES.keys(), default="distilbert")
    parser.add_argument("--epochs", type=int, default=4)
    args = parser.parse_args()

    model_name = MODEL_CHOICES[args.model]
    print(f"Fine-tuning {model_name} ...")

    texts, labels = load_data()
    print(f"Loaded {len(texts)} labeled clauses.")

    # SAME split logic as train_classifier.py (random_state=42, stratified)
    # so the two legs' test sets are directly comparable.
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_enc = tokenizer(X_train, truncation=True, padding=True, max_length=128, return_tensors="pt")
    test_enc = tokenizer(X_test, truncation=True, padding=True, max_length=128, return_tensors="pt")

    train_ds = ClauseDataset(train_enc, y_train)
    test_ds = ClauseDataset(test_enc, y_test)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(CATEGORY_IDS), id2label=ID2LABEL, label2id=LABEL2ID,
    )

    training_args = TrainingArguments(
        output_dir=str(MODEL_OUT_DIR / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=5,
        learning_rate=2e-5,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    final_metrics = trainer.evaluate()
    print("\nFinal eval metrics:", final_metrics)

    preds = trainer.predict(test_ds)
    y_pred = np.argmax(preds.predictions, axis=-1)
    print("\nFull classification report (same test set as TF-IDF baseline):")
    print(classification_report(y_test, y_pred, target_names=list(ID2LABEL.values()), zero_division=0))

    MODEL_OUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(MODEL_OUT_DIR)
    tokenizer.save_pretrained(MODEL_OUT_DIR)
    print(f"Saved fine-tuned model to {MODEL_OUT_DIR}")


if __name__ == "__main__":
    main()
