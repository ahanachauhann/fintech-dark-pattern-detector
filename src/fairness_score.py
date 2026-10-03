"""
Aggregates per-clause dark-pattern predictions into a single "Fairness
Score" per document/product. Simple, explainable scoring — start here,
refine the weighting later if you want.
"""
from categories import CATEGORIES

# Weight each category by how consumer-harmful it tends to be — you can
# tune these yourself and justify the reasoning in your report.
CATEGORY_WEIGHTS = {
    "hidden_fee": 2.0,
    "auto_renewal_trap": 1.5,
    "penalty_obscurity": 1.5,
    "deceptive_framing": 1.0,
    "forced_arbitration": 2.0,
    "none": 0.0,
}


def compute_fairness_score(clause_labels: list[str]) -> dict:
    """
    clause_labels: list of category ids, one per clause in a document.
    Returns a dict with the raw penalty total, a normalized 0-100 score
    (100 = fairest / no dark patterns), and a breakdown by category.
    """
    total_clauses = len(clause_labels) or 1
    breakdown = {}
    penalty = 0.0
    for cid in CATEGORIES:
        count = clause_labels.count(cid)
        breakdown[cid] = count
        penalty += count * CATEGORY_WEIGHTS.get(cid, 1.0)

    # Normalize: more clauses flagged relative to document length = lower score
    max_possible = total_clauses * max(CATEGORY_WEIGHTS.values())
    normalized = 100 * (1 - min(penalty / max_possible, 1.0)) if max_possible else 100

    return {
        "fairness_score": round(normalized, 1),
        "raw_penalty": round(penalty, 1),
        "total_clauses": total_clauses,
        "breakdown": breakdown,
    }
