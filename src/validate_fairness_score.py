"""
Does the Fairness Score mean anything? Scores each document two ways and compares them:
  true score      : from the hand-verified (v3) clause labels
  predicted score : from a model that NEVER saw that document (leave-one-document-out)
Also checks how much the arbitrary category weights matter.

Usage: python src/validate_fairness_score.py     (writes results/fairness_validation.txt)
"""
import csv
import os

import numpy as np
from scipy.stats import spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

import fairness_score as fs
from cv_splits import EXCLUDE_DOCS, ROOT, TAG


def main():
    path = ROOT / "data" / "labeled" / os.environ.get("DP_LABELS", "clauses_reviewed_v3.csv")
    rows = [r for r in csv.DictReader(open(path, encoding="utf-8")) if r["label"]]
    rows = [r for r in rows if r["source_file"] not in EXCLUDE_DOCS]   # DP_EXCLUDE robustness check
    texts = [r["clause_text"] for r in rows]
    y = np.array([r["label"] for r in rows])
    g = np.array([r["source_file"] for r in rows])
    docs = sorted(set(g))

    pred = np.empty(len(rows), dtype=object)
    for d in docs:
        te = g == d
        vec = TfidfVectorizer(max_features=2000, ngram_range=(1, 2), stop_words="english")
        Xtr = vec.fit_transform([t for t, m in zip(texts, ~te) if m])
        clf = LogisticRegression(max_iter=1000, class_weight="balanced").fit(Xtr, y[~te])
        pred[te] = clf.predict(vec.transform([t for t, m in zip(texts, te) if m]))

    lines = [f"Fairness Score validation | labels: {path.name} | {len(docs)} documents", "",
             f"{'document':32s} true_score  predicted_score"]
    true_s, pred_s = [], []
    for d in docs:
        m = g == d
        a = fs.compute_fairness_score(list(y[m]))["fairness_score"]
        b = fs.compute_fairness_score(list(pred[m]))["fairness_score"]
        true_s.append(a)
        pred_s.append(b)
        lines.append(f"{d:32s} {a:8.1f}  {b:10.1f}")
    r = spearmanr(true_s, pred_s)
    lines += ["", f"Spearman rank correlation, hand-label score vs unseen-document model score: "
                  f"{r.correlation:.2f} (p={r.pvalue:.3f})",
              f"Mean absolute gap: {np.mean(np.abs(np.array(true_s) - np.array(pred_s))):.1f} points"]

    original = dict(fs.CATEGORY_WEIGHTS)
    fs.CATEGORY_WEIGHTS = {k: (0.0 if k == "none" else 1.0) for k in original}
    equal = [fs.compute_fairness_score(list(y[g == d]))["fairness_score"] for d in docs]
    fs.CATEGORY_WEIGHTS = original
    lines.append(f"Ranking agreement, current weights vs all-equal weights: Spearman {spearmanr(true_s, equal).correlation:.2f}")
    report = "\n".join(lines)
    print(report)
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / f"fairness_validation{TAG}.txt").write_text(report + "\n")


if __name__ == "__main__":
    main()
