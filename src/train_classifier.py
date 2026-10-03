"""
Trains a simple, explainable classifier: TF-IDF (turns text into numeric
features based on word importance) + Logistic Regression (a standard,
easy-to-explain classifier). This is intentionally the "easy mode" model —
upgrade to a fine-tuned transformer later only if time allows.

Usage:
    python src/train_classifier.py
"""
import csv
import pickle
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import classification_report

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "labeled" / "clauses_reviewed.csv"
MODEL_OUT = Path(__file__).resolve().parent.parent / "data" / "labeled" / "classifier.pkl"


def load_data():
    texts, labels = [], []
    with open(DATA_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["label"]:
                texts.append(row["clause_text"])
                labels.append(row["label"])
    return texts, labels


def main():
    texts, labels = load_data()
    print(f"Loaded {len(texts)} labeled clauses.")

    vectorizer = TfidfVectorizer(max_features=2000, ngram_range=(1, 2), stop_words="english")
    X = vectorizer.fit_transform(texts)

    # Small dataset — cross-validation gives a more honest performance
    # estimate than a single train/test split. Report this range in your
    # writeup rather than a single accuracy number.
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    scores = cross_val_score(clf, X, labels, cv=5)
    print(f"Cross-validated accuracy: {scores.mean():.2f} (+/- {scores.std():.2f})")

    # Fit on everything for the final deployed model, but also print a
    # held-out report for your report's appendix.
    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=0.2, random_state=42, stratify=labels
    )
    clf.fit(X_train, y_train)
    print(classification_report(y_test, clf.predict(X_test)))

    clf.fit(X, labels)  # final fit on all data
    with open(MODEL_OUT, "wb") as f:
        pickle.dump({"vectorizer": vectorizer, "classifier": clf}, f)
    print(f"Saved model to {MODEL_OUT}")


if __name__ == "__main__":
    main()
