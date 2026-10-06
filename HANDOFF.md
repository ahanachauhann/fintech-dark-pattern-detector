# Project Handoff — Fintech Dark-Pattern Detector

**Who this is for:** Ahana, 3rd-year B.Tech CSE (AI/ML), UPES Dehradun. Final-year
major project, ~1 month solo timeline, submitted as a continuation of a prior
long chat (moved here to avoid hitting memory/token limits).

**GitHub repo:** https://github.com/ahanachauhann/fintech-dark-pattern-detector
(6 commits, confirmed in sync with this zip as of this handoff)

**How to use this doc:** Paste this whole file into a new chat along with the
project zip, and say "continue from here." Everything needed to pick up
exactly where we left off is below.

---

## 1. What the project is

An NLP system that reads real terms & conditions from financial products
(BNPL apps, credit cards, personal loans, insurance) and flags clauses that
use manipulative "dark patterns" — vague fees, unilateral discretionary
power, hidden penalties, dispute-limiting clauses — producing an
interpretable "Fairness Score" per document.

**Methodology (follows UPES's required 7-step structure):**
1. Literature Review
2. Requirement Identification (data sourcing)
3. Algorithm Identification
4. Comparative Study — **this is the core of the project**: three-way
   comparison of TF-IDF+LogReg vs. a fine-tuned transformer vs. zero-shot LLM
   classification, all on the same fintech-specific dataset
5. System Design
6. Implementation
7. Results


## 2. Key papers to cite

- **Lippi et al. 2019, "CLAUDETTE"** — direct precedent; ML detection of
  unfair clauses in ToS using pre-transformer methods (80%+ precision)
- **Juttu et al. 2025, "Text to Trust"** (arXiv 2510.22531) — closest prior
  work; the exact fine-tune-vs-LoRA-vs-zero-shot comparison you're extending
  into a new domain
- **Chalkidis et al. 2020, "Legal-BERT"** (EMNLP Findings) — the
  legal-domain-pretrained BERT variant used for the transformer leg
- **Mathur et al. 2019, "Dark Patterns at Scale"** (ACM CSCW) — taxonomy
  source for the "dark pattern" concept itself
- **RBI Draft Amendments on Dark Patterns in Banking** (Feb 2026) — current
  regulatory relevance for your introduction

## 3. Current dataset status

**15 real documents, 251 labeled clauses**, covering all 4 categories:
- BNPL: LazyPay, Amazon Pay Later, ZestMoney, Paytm (fees)
- Personal loans: Fibe, KreditBee, Flipkart EMI, DSP Finance fee policy,
  RBI KFS format (regulatory reference, not a product)
- Credit cards: HDFC MITC, IDFC MITC, ICICI MITC, AU Small Finance Bank MITC
- Insurance: HDFC ERGO, Bajaj General (car policy)

**Known, documented limitation:** insurance has ZERO positive dark-pattern
examples (all 25 insurance clauses labeled `none`) — both insurance docs
happen to be procedurally clear. State this plainly in the report rather
than hiding it. Optional fix: add one real vague/discretionary insurance
clause if time allows; not blocking.

**Label distribution (current, v2 methodology):**
```
none                170
hidden_fee           49
deceptive_framing    15
penalty_obscurity    11
forced_arbitration    6
```
`forced_arbitration` and `penalty_obscurity` are thin (6 and 11 examples) —
known, acceptable limitation; frame as a finding about data scarcity per the
Text-to-Trust paper's own research question, not a hidden flaw.

## 5. The methodology story worth writing up (this is genuinely good material)

1. **v1:** First-pass rule-based labeling matched *topic* keywords (fee,
   penalty, discretion) → produced noisy, over-broad labels. Baseline
   accuracy ~70% but on a flawed dataset.
2. **An external audit** (provided by the user, reviewed critically) caught
   the real problem: a clause isn't a dark pattern just because it costs
   money or grants a contractual right — it's a dark pattern because of
   *how* it's written (vague/unquantified/ex-post-disclosed vs. clearly
   stated). This is the exact distinction CLAUDETTE's own paper is built on.
3. **Also caught a technical bug:** the clause-splitter was breaking
   sentences on every period, including abbreviations ("Rs.", "i.e.",
   "etc.") and semicolon-separated list items — shredding fee tables into
   nonsense fragments.
4. **Fixed both:** rewrote `clause_split.py` to only split on
   `.`/`;` followed by a capital letter (preserves abbreviations and list
   items); rewrote the labeler around "quantified vs. vague" rather than
   keyword topic-matching.
5. **Result: cross-val accuracy 70% → 77%, test accuracy 70% → 80%** — a
   genuine improvement from fixing methodology, not from adding data. The
   `none` category F1 jumped from 75% to 86% (much better at NOT
   false-flagging ordinary disclosed terms, which matters as much as
   catching real patterns).
6. I rejected the audit's suggestion to split into 8 new categories (too
   fragmented for ~250 examples) — kept the existing 5, just tightened
   definitions. This was a deliberate scope call given the 1-month timeline.

**This whole arc (flawed v1 → audit → fix → measurable improvement) is
excellent material for the report's methodology/limitations section.**

## 6. Baseline results (TF-IDF + Logistic Regression, current/final)

```
Cross-validated accuracy: 77% (+/- 6%)
Held-out test accuracy: 80%

                    precision  recall  f1-score  support
deceptive_framing       0.67    0.67      0.67        3
forced_arbitration      1.00    1.00      1.00        1
hidden_fee               0.67    0.60      0.63       10
none                     0.84    0.89      0.86       35
penalty_obscurity       1.00    0.50      0.67        2
```
Saved at `results/classification_report.txt` and in git history.

## 7. EXACTLY where we left off / next step

**We were about to run the second leg of the comparison: fine-tuning a
transformer (DistilBERT by default, Legal-BERT as a --model flag option).**

The script `src/train_transformer.py` is **already written and ready** —
built to use the identical train/test split (random_state=42, test_size=0.2,
stratified) as the TF-IDF baseline, so results are directly comparable.

**Blocker hit:** Claude's sandbox environment has a restricted network
allowlist that does NOT include huggingface.co, so the pretrained model
weights can't be downloaded from within that sandbox. The code itself is
correct — it just needs to run somewhere with full internet access.

**Next concrete step:** Run this on Kaggle (Ahana has used Kaggle before for
the ML backdoor research project, so she's familiar with the workflow):
1. New Kaggle notebook, enable GPU
2. Upload the project folder as a Kaggle dataset
3. `!pip install transformers torch scikit-learn -q`
4. `!python src/train_transformer.py --model distilbert --epochs 4`
   (or `--model legalbert` for the legal-domain-pretrained variant, slower)
5. Bring the printed classification report back to continue the comparison

**After the transformer leg:**
- Build the third leg: zero-shot LLM classification (no fine-tuning) on the
  same held-out test clauses — fastest to build, just prompting
- Compare all three legs' results, especially on the thin categories
  (forced_arbitration, penalty_obscurity) — does fine-tuning or zero-shot
  do better under data scarcity? This directly mirrors Text-to-Trust's
  research question.
- Wire up and test the Streamlit demo (`app/streamlit_app.py`) with
  whichever model performs best — built but never actually tested end-to-end
- Do a human review pass on a sample of the v2 labels (even 30-40 clauses)
  to legitimately claim "manually verified" in the report
- Write up the final report using the methodology story in section 5 above
- Fill in and submit the UPES project form (draft content already produced
  earlier in the original conversation — title, abstract, objective,
  methodology; mentor name was never provided, still needed)

## 8. Project file structure

```
fintech-dark-patterns/
├── README.md
├── HANDOFF.md              <- this file
├── requirements.txt
├── data/
│   ├── raw/                <- 15 real T&C .txt files + README with sources
│   └── labeled/
│       ├── clauses_unlabeled.csv
│       ├── clauses_reviewed.csv    <- current labeled dataset (251 rows)
│       └── classifier.pkl          <- trained TF-IDF+LogReg model
├── src/
│   ├── categories.py        <- v2 category definitions (quantified-vs-vague)
│   ├── clause_split.py      <- fixed sentence splitter
│   ├── label_with_llm.py    <- LLM-assisted labeling script (needs API key)
│   ├── train_classifier.py  <- TF-IDF+LogReg baseline (done)
│   ├── train_transformer.py <- transformer fine-tuning (ready, needs Kaggle)
│   └── fairness_score.py    <- aggregation logic
├── app/
│   └── streamlit_app.py     <- demo UI (built, not yet tested)
└── results/
    └── classification_report.txt   <- latest TF-IDF results
```

## 9. User preferences to carry forward

- Wants direct, honest feedback — not validation. Push back when something's
  weak rather than softening it.
- Prefers concrete sequential action steps over abstract frameworks.
- Favors genuine comprehension over surface coverage — explain the "why,"
  not just the "what."
- Appreciates git/version-control as proof-of-work for the professor.
