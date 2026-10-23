"""
Demo UI: paste in terms & conditions text, see flagged clauses and a Fairness Score.

Run with:
    streamlit run app/streamlit_app.py
"""
import pickle
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
from categories import CATEGORIES
from clause_split import split_into_clauses
from fairness_score import compute_fairness_score

MODEL_PATH = Path(__file__).resolve().parent.parent / "data" / "labeled" / "classifier.pkl"

st.set_page_config(page_title="Fintech Dark-Pattern Detector", layout="wide")
st.title("Fintech Dark-Pattern Detector")
st.caption("Paste in terms & conditions text to flag potentially manipulative clauses.")

if not MODEL_PATH.exists():
    st.warning("No trained model found yet. Run `python src/train_classifier.py` first.")
    st.stop()


@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        saved = pickle.load(f)
    return saved["vectorizer"], saved["classifier"]


vectorizer, clf = load_model()

text = st.text_area("Paste terms & conditions text here", height=250)

if st.button("Analyze") and text.strip():
    clauses = split_into_clauses(text)
    if not clauses:
        st.info("No clauses were detected. Try pasting longer or differently formatted text.")
        st.stop()

    proba = clf.predict_proba(vectorizer.transform(clauses))
    best = proba.argmax(axis=1)
    labels = [str(clf.classes_[i]) for i in best]
    confidence = proba.max(axis=1)

    result = compute_fairness_score(labels)
    flagged = [(c, l, p) for c, l, p in zip(clauses, labels, confidence) if l != "none"]

    left, right = st.columns(2)
    left.metric("Fairness Score", f"{result['fairness_score']} / 100")
    right.metric("Clauses flagged", f"{len(flagged)} of {len(clauses)}")

    st.subheader("Breakdown by pattern")
    rows = [{"Pattern": CATEGORIES[cid]["label"], "Clauses": n}
            for cid, n in result["breakdown"].items() if cid != "none" and n > 0]
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True)
    else:
        st.write("No dark-pattern clauses were flagged.")

    st.subheader("Flagged clauses")
    for clause, label, conf in sorted(flagged, key=lambda t: -t[2]):
        info = CATEGORIES[label]
        st.markdown(f"**{info['label']}**  (model confidence {conf:.0%})")
        st.write(clause)
        st.caption(info["description"])
        st.divider()

with st.expander("How this works, and its limits"):
    st.write(
        "Each clause is classified by a TF-IDF + logistic regression model trained on 251 hand-reviewed "
        "clauses from 15 Indian fintech documents (loans, BNPL, credit cards, insurance). The Fairness Score "
        "is 100 minus a weighted share of flagged clauses. This is a research prototype: it can miss patterns, "
        "can flag harmless clauses, and has very little training data for fee-related and arbitration patterns. "
        "Treat the output as a prompt for human review, not as legal advice."
    )
