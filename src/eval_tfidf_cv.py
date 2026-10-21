"""
TF-IDF + Logistic Regression scored with repeated cross-validation on the
shared folds from cv_splits.py. Does NOT overwrite classifier.pkl.

Usage:
    python src/eval_tfidf_cv.py                  # stratified (random clauses)
    python src/eval_tfidf_cv.py --mode grouped   # hold out whole documents
"""
import argparse
import json
from collections import defaultdict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score

from cv_splits import ROOT, get_splits, load_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["stratified", "grouped"], default="stratified")
    args = ap.parse_args()

    rows = load_rows()
    texts = [r["clause_text"] for r in rows]
    y = np.array([r["label"] for r in rows])
    labels = sorted(set(y))
    splits = get_splits(rows, args.mode)

    accs, macros = [], []
    per_class = defaultdict(list)
    for tr, te in splits:
        vec = TfidfVectorizer(max_features=2000, ngram_range=(1, 2), stop_words="english")
        Xtr = vec.fit_transform([texts[i] for i in tr])   # fit on TRAIN only
        Xte = vec.transform([texts[i] for i in te])
        clf = LogisticRegression(max_iter=1000, class_weight="balanced").fit(Xtr, y[tr])
        pred = clf.predict(Xte)
        accs.append(accuracy_score(y[te], pred))
        macros.append(f1_score(y[te], pred, labels=labels, average="macro", zero_division=0))
        f1s = f1_score(y[te], pred, labels=labels, average=None, zero_division=0)
        for lab, f in zip(labels, f1s):
            if (y[te] == lab).any():          # only score a class on folds where it exists
                per_class[lab].append(f)

    lines = [f"TF-IDF + LogReg | mode={args.mode} | {len(rows)} clauses | {len(splits)} folds",
             f"accuracy : {np.mean(accs):.3f} +/- {np.std(accs):.3f}",
             f"macro-F1 : {np.mean(macros):.3f} +/- {np.std(macros):.3f}",
             "per-class F1 (mean +/- std over folds where the class appears):"]
    for lab in labels:
        v = per_class[lab]
        lines.append(f"  {lab:20s} {np.mean(v):.3f} +/- {np.std(v):.3f}  (n_folds={len(v)})")
    report = "\n".join(lines)
    print(report)

    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    (out / f"tfidf_cv_{args.mode}.txt").write_text(report + "\n")
    json.dump({"accuracy": accs, "macro_f1": macros}, open(out / f"tfidf_cv_{args.mode}.json", "w"))


if __name__ == "__main__":
    main()
