# Handoff: label verification finished, v3 dataset built (for a new chat)

Repo: github.com/ahanachauhann/fintech-dark-pattern-detector | local: /Volumes/New Volume/fintech-dark-patterns
Git tag `labels-v2-unverified` = original labels + all pre-review results (TF-IDF, DistilBERT run 2, Qwen zero-shot).

## What was done
- 40-clause blind review (25 random + 15 flagged): 52% raw / 59% prevalence-weighted agreement with v2 labels, kappa 0.39.
- Targeted relabel of 76 clauses (chosen by type/keywords, not by model predictions) under refined rules (see review/GUIDE.txt + rules below).
- data/labeled/clauses_reviewed_v3.csv = v2 + hand corrections (115 reviewed rows; 79 changed; 135 rows unreviewed, unchanged).
  v2 -> v3 counts: none 170->177, deceptive_framing 15->49, hidden_fee 49->15, penalty_obscurity 11->5, forced_arbitration 6->5.
  Change log: results/relabel_changelog.csv.
- Scripts now accept DP_LABELS=clauses_reviewed_v3.csv and DP_TAG=_v3 (result files get the suffix; v2 results untouched).
- Macro-F1 now averaged over classes present in each test fold (old code scored absent classes as 0, which deflated grouped macro-F1 in v2).
  Old v2 grouped numbers (TF-IDF 0.519, DistilBERT 0.573, Qwen 0.363) used the old definition; stratified v2 numbers are unaffected.

## Rules used for v3
none: consumer-protective/disclosure rules, fully stated terms, plain info, standard disclaimers, ordinary default consequences.
hidden_fee: fee/charge whose amount, rate or trigger/threshold is not stated; "may vary"; charges outside headline rate.
penalty_obscurity: only late/default MONEY penalty with unspecified amount/trigger.
deceptive_framing: unilateral power (sole discretion, without notice/reason, change/reject/suspend/disclose at will); discretion over a charge counts here.
forced_arbitration: arbitration, waiver of dispute rights, deemed-acceptance windows; a bank's own "final and binding decision" = deceptive_framing.

## Next steps
1. Rerun on v3 with the SAME frozen folds (results/cv_splits.json): TF-IDF (local), DistilBERT (Kaggle, CUDA_VISIBLE_DEVICES=0, 20 epochs lr 5e-5 batch 8), zero-shot Qwen2.5-7B (Kaggle, both GPUs).
   Prefix every command with: DP_LABELS=clauses_reviewed_v3.csv DP_TAG=_v3
2. Compare: DP_TAG=_v3 python compare_cv.py tfidf distilbert grouped (etc.).
3. Small classes under v3 (penalty_obscurity 5, forced_arbitration 5, hidden_fee 15): per-class results very noisy; some folds have zero test examples.
   Consider an additional merged analysis (none / unilateral control = deceptive+forced / fee opacity = hidden_fee+penalty) -- needs a DP_MERGE option.
4. Report v2 and v3 side by side with label-reliability numbers. Remaining: Streamlit demo test, report, UPES form (mentor name).
