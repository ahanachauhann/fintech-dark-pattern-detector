"""
Fine-tunes a transformer on the SAME repeated-CV folds as eval_tfidf_cv.py
(read from results/cv_splits.json), so TF-IDF vs transformer is a fair,
fold-for-fold comparison. Meant to run on a GPU (Kaggle).

Fairness notes:
  * TF-IDF used class_weight="balanced"; here the loss is class-weighted too,
    otherwise the transformer just learns to say "none".
  * A fresh pretrained model is loaded for every fold (no leakage between folds).
  * Folds are only LOADED, never regenerated (different library versions would
    produce different folds and break comparability).

Usage (smoke test first!):
    python src/train_transformer_cv.py --model distilbert --mode stratified --max-folds 1 --epochs 1
Full run:
    python src/train_transformer_cv.py --model distilbert --mode stratified
    python src/train_transformer_cv.py --model distilbert --mode grouped
    python src/train_transformer_cv.py --model legalbert  --mode grouped
"""
import argparse
import gc
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import Dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          Trainer, TrainingArguments, set_seed)

from cv_splits import ROOT, SPLITS_PATH, TAG, apply_exclusion, load_rows

MODEL_CHOICES = {
    "distilbert": "distilbert-base-uncased",
    "legalbert": "nlpaueb/legal-bert-base-uncased",
}


class ClauseDataset(Dataset):
    def __init__(self, enc, labels):
        self.enc, self.labels = enc, labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        item = {k: v[i] for k, v in self.enc.items()}
        item["labels"] = torch.tensor(self.labels[i])
        return item


class WeightedTrainer(Trainer):
    def __init__(self, class_weights, **kw):
        super().__init__(**kw)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        w = self.class_weights.to(outputs.logits.device)
        loss = torch.nn.functional.cross_entropy(outputs.logits, labels, weight=w)
        return (loss, outputs) if return_outputs else loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=MODEL_CHOICES, default="distilbert")
    ap.add_argument("--mode", choices=["stratified", "grouped"], default="stratified")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--batch-size", type=int, default=8, help="per-device batch size")
    ap.add_argument("--warmup", type=float, default=0.1, help="fraction of steps used for LR warmup")
    ap.add_argument("--max-folds", type=int, default=None, help="smoke test: only run first N folds")
    args = ap.parse_args()

    if not SPLITS_PATH.exists():
        raise SystemExit(f"{SPLITS_PATH} missing - pull the repo; do NOT regenerate folds here.")
    splits = json.loads(SPLITS_PATH.read_text())[args.mode]
    if args.max_folds:
        splits = splits[: args.max_folds]

    rows = load_rows()
    splits = apply_exclusion(splits, rows)
    texts = [r["clause_text"] for r in rows]
    labels_all = sorted({r["label"] for r in rows})
    l2i = {l: i for i, l in enumerate(labels_all)}
    i2l = {i: l for l, i in l2i.items()}
    y = np.array([l2i[r["label"]] for r in rows])
    name = MODEL_CHOICES[args.model]
    n_dev = max(1, torch.cuda.device_count())
    print(f"{name} | mode={args.mode} | {len(rows)} clauses | {len(splits)} folds | gpus={n_dev} | effective batch={args.batch_size * n_dev}")

    accs, macros, train_f1s = [], [], []
    per_class = defaultdict(list)
    tok = AutoTokenizer.from_pretrained(name)

    for k, (tr, te) in enumerate(splits):
        set_seed(42 + k)
        tr_texts, te_texts = [texts[i] for i in tr], [texts[i] for i in te]
        tr_enc = tok(tr_texts, truncation=True, padding=True, max_length=128, return_tensors="pt")
        te_enc = tok(te_texts, truncation=True, padding=True, max_length=128, return_tensors="pt")
        counts = np.bincount(y[tr], minlength=len(labels_all)).astype(float)
        weights = torch.tensor(counts.sum() / (len(labels_all) * np.maximum(counts, 1)), dtype=torch.float)

        model = AutoModelForSequenceClassification.from_pretrained(
            name, num_labels=len(labels_all), id2label=i2l, label2id=l2i)
        steps = int(np.ceil(len(tr) / (args.batch_size * n_dev))) * args.epochs
        targs = TrainingArguments(
            output_dir=str(ROOT / "tmp_cv"), num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size, warmup_steps=int(args.warmup * steps), per_device_eval_batch_size=32,
            learning_rate=args.lr, weight_decay=0.01, save_strategy="no",
            logging_strategy="no", report_to=[], seed=42 + k,
            fp16=torch.cuda.is_available(), disable_tqdm=True)
        trainer = WeightedTrainer(class_weights=weights, model=model, args=targs,
                                  train_dataset=ClauseDataset(tr_enc, y[tr].tolist()))
        trainer.train()
        pred = np.argmax(trainer.predict(ClauseDataset(te_enc, y[te].tolist())).predictions, axis=-1)

        tr_pred = np.argmax(trainer.predict(ClauseDataset(tr_enc, y[tr].tolist())).predictions, axis=-1)
        idx = list(range(len(labels_all)))
        train_f1s.append(f1_score(y[tr], tr_pred, labels=idx, average="macro", zero_division=0))
        accs.append(accuracy_score(y[te], pred))
        macros.append(f1_score(y[te], pred, labels=sorted(set(y[te])), average="macro", zero_division=0))
        f1s = f1_score(y[te], pred, labels=idx, average=None, zero_division=0)
        for i, f in zip(idx, f1s):
            if (y[te] == i).any():
                per_class[i2l[i]].append(f)
        print(f"fold {k+1}/{len(splits)}: steps={steps} train macroF1={train_f1s[-1]:.3f} | test acc={accs[-1]:.3f} macroF1={macros[-1]:.3f}", flush=True)

        del trainer, model
        gc.collect()
        torch.cuda.empty_cache()

    lines = [f"{name} | mode={args.mode} | {len(rows)} clauses | {len(splits)} folds | epochs={args.epochs} lr={args.lr} batch={args.batch_size*n_dev}",
             f"accuracy : {np.mean(accs):.3f} +/- {np.std(accs):.3f}",
             f"macro-F1 : {np.mean(macros):.3f} +/- {np.std(macros):.3f}",
             f"train macro-F1 (fit check, should be high): {np.mean(train_f1s):.3f}",
             "per-class F1 (mean +/- std over folds where the class appears):"]
    for lab in labels_all:
        v = per_class[lab]
        lines.append(f"  {lab:20s} {np.mean(v):.3f} +/- {np.std(v):.3f}  (n_folds={len(v)})")
    report = "\n".join(lines)
    print("\n" + report)

    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    tag = f"{args.model}_cv_{args.mode}{TAG}" + (f"_smoke{args.max_folds}" if args.max_folds else "")
    (out / f"{tag}.txt").write_text(report + "\n")
    json.dump({"accuracy": accs, "macro_f1": macros, "train_macro_f1": train_f1s}, open(out / f"{tag}.json", "w"))


if __name__ == "__main__":
    main()
