# Fintech Dark-Pattern Detector

Reads real terms & conditions from loan apps, BNPL apps, credit cards, and
insurance policies, and flags clauses that use manipulative/predatory
design patterns ("dark patterns"). Produces a "Fairness Score" per product.

## Project structure

```
fintech-dark-patterns/
├── data/
│   ├── raw/           # raw T&C text files, one per product
│   └── labeled/        # clause-level labeled dataset (CSV)
├── src/
│   ├── categories.py    # dark-pattern category definitions
│   ├── clause_split.py  # splits raw T&C text into clause-level rows
│   ├── label_with_llm.py# first-pass labeling using an LLM API
│   ├── train_classifier.py # trains TF-IDF + Logistic Regression classifier
│   └── fairness_score.py   # aggregates clause predictions into a score
├── app/
│   └── streamlit_app.py # simple demo UI
└── requirements.txt
```


## Setup

```bash
pip install -r requirements.txt
```

You'll need an API key for `label_with_llm.py` (Anthropic or OpenAI) —
set it as an environment variable, see the top of that file.

## Important note on the Fairness Score

Consider anonymizing or grouping real company names (e.g. "Product A/B/C")
in anything you publish or present publicly, rather than a named public
ranking of real companies — worth a deliberate decision, not a default.
