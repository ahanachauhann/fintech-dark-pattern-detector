"""
Dark-pattern categories for the fintech contract detector.

Adjust these definitions as you refine your project — these are a
starting point based on well-documented consumer-protection concerns
in lending/BNPL/insurance products.
"""

CATEGORIES = {
    "hidden_fee": {
        "label": "Hidden / unclear fee",
        "description": (
            "A charge, fee, or deduction that is not clearly stated upfront, "
            "or is buried in dense unrelated text rather than disclosed "
            "prominently."
        ),
        "example_cue_words": ["processing fee", "deducted from", "additional charges", "GST"],
    },
    "auto_renewal_trap": {
        "label": "Auto-renewal / hard-to-cancel",
        "description": (
            "Terms that auto-renew a subscription or credit line without "
            "clear opt-out, or make cancellation deliberately difficult."
        ),
        "example_cue_words": ["automatically renew", "unless cancelled", "continue until terminated"],
    },
    "penalty_obscurity": {
        "label": "Confusing penalty / default clause",
        "description": (
            "Penalty, late-fee, or default terms written in a way that "
            "obscures the real cost or consequence to the user."
        ),
        "example_cue_words": ["penalty", "late payment", "default", "overdue"],
    },
    "deceptive_framing": {
        "label": "Deceptive framing / risk minimization",
        "description": (
            "Language that frames a risky or costly term in a way that "
            "downplays it, or uses vague reassurance instead of clear facts."
        ),
        "example_cue_words": ["at its sole discretion", "may vary", "as applicable"],
    },
    "forced_arbitration": {
        "label": "Forced arbitration / dispute-limiting clause",
        "description": (
            "Terms that limit the user's ability to dispute decisions, "
            "waive rights to legal recourse, or force one-sided arbitration."
        ),
        "example_cue_words": ["sole discretion", "final and binding", "waive", "arbitration"],
    },
    "none": {
        "label": "No dark pattern detected",
        "description": "A neutral, clearly disclosed clause with no manipulative pattern.",
        "example_cue_words": [],
    },
}

CATEGORY_IDS = list(CATEGORIES.keys())
