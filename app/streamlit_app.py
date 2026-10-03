"""
Simple demo UI. Paste in T&C text, see flagged clauses and a Fairness Score.

Run with:
    streamlit run app/streamlit_app.py
"""
import pickle
import sys
from pathlib import Path

import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
from categories import CATEGORIES
from clause_split import split_into_clauses
from fairness_score import compute_fairness_score

MODEL_PATH = Path(__file__).resolve().parent.parent / "data" / "labeled" / "classifier.pkl"

st.title("Fintech Dark-Pattern Detector")
st.caption("Paste in terms & conditions text to flag manipulative clauses.")

if not MODEL_PATH.exists():
    st.warning("No trained model found yet — run train_classifier.py first (Week 3).")
    st.stop()

with open(MODEL_PATH, "rb") as f:
    saved = pickle.load(f)
vectorizer, clf = saved["vectorizer"], saved["classifier"]

text = st.text_area("Paste terms & conditions text here", height=250)

if st.button("Analyze") and text.strip():
    clauses = split_into_clauses(text)
    X = vectorizer.transform(clauses)
    predictions = clf.predict(X)

    result = compute_fairness_score(list(predictions))
    st.metric("Fairness Score", f"{result['fairness_score']} / 100")

    st.subheader("Flagged clauses")
    for clause, label in zip(clauses, predictions):
        if label != "none":
            info = CATEGORIES[label]
            st.markdown(f"**{info['label']}**")
            st.write(clause)
            st.caption(info["description"])
            st.divider()
