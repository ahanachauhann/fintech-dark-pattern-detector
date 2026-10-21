"""
Shared cross-validation splits for ALL three legs of the comparison
(TF-IDF, fine-tuned transformer, zero-shot LLM). Generating the folds once
and saving them to results/cv_splits.json guarantees every method is scored
on exactly the same train/test clauses.

Two modes:
  stratified : clauses split at random, class balance preserved.
               Easier; clauses from the same document can land in both
               train and test (boilerplate leaks across the split).
  grouped    : whole DOCUMENTS are held out together (by source_file).
               Harder and closer to the real use case - scoring a document
               the model has never seen.
"""
import csv
import json
from pathlib import Path

from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedGroupKFold

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "labeled" / "clauses_reviewed.csv"
SPLITS_PATH = ROOT / "results" / "cv_splits.json"


def load_rows():
    with open(DATA_PATH, newline="", encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if r["label"]]


def make_splits(rows, mode="stratified", n_splits=5, n_repeats=3, seed=42):
    y = [r["label"] for r in rows]
    X = list(range(len(rows)))
    splits = []
    if mode == "stratified":
        cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
        for tr, te in cv.split(X, y):
            splits.append((tr.tolist(), te.tolist()))
    elif mode == "grouped":
        groups = [r["source_file"] for r in rows]
        for rep in range(n_repeats):
            cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed + rep)
            for tr, te in cv.split(X, y, groups):
                splits.append((tr.tolist(), te.tolist()))
    else:
        raise ValueError(mode)
    return splits


def save_splits(splits, mode):
    SPLITS_PATH.parent.mkdir(exist_ok=True)
    existing = json.loads(SPLITS_PATH.read_text()) if SPLITS_PATH.exists() else {}
    existing[mode] = splits
    SPLITS_PATH.write_text(json.dumps(existing))


def get_splits(rows, mode="stratified"):
    """Load saved folds if present (so every method reuses them), else create and save."""
    if SPLITS_PATH.exists():
        saved = json.loads(SPLITS_PATH.read_text())
        if mode in saved and len(saved[mode]) and max(max(t) for _, t in saved[mode]) < len(rows):
            return [(tr, te) for tr, te in saved[mode]]
    splits = make_splits(rows, mode)
    save_splits(splits, mode)
    return splits
