"""
Dark-pattern categories for the fintech contract detector.

CORE LABELING PRINCIPLE (v2, revised after human audit of 160
auto-labeled clauses in week 2):

    A clause is a dark-pattern candidate only if the cost, power, or
    consequence it describes is VAGUE, UNQUANTIFIED, DISCRETIONARY, or
    UNDISCLOSED UNTIL AFTER THE FACT -- not merely because the company
    is charging something, imposing a penalty, or holding a normal
    contractual right.

    A clearly quantified fee ("2.5% subject to a minimum of Rs 500")
    is NOT hidden_fee. A fee described with no amount ("a convenience
    fee may be chargeable") IS. The same rule applies to
    penalty_obscurity: a penalty with an explicit table is NOT
    obscure; a penalty "as may be determined at the Bank's sole
    discretion" IS.

This matters for the classifier: without this rule, the model learns
to flag "contains financial vocabulary" rather than "contains a dark
pattern" -- the two are not the same, and conflating them was an
earlier, now-corrected version of this taxonomy.
"""

CATEGORIES = {
    "hidden_fee": {
        "label": "Hidden / unquantified fee",
        "description": (
            "A charge or fee whose AMOUNT, RATE, OR TRIGGER IS NOT GIVEN "
            "in the clause itself (vague: 'may be chargeable', 'a nominal "
            "fee', 'as applicable', amount set elsewhere/later), OR a fee "
            "only disclosed to the user AFTER the transaction (e.g. "
            "'visible in your subsequent statement'). A clause that states "
            "a specific percentage, amount, or capped figure is NOT this "
            "category -- it is a disclosed fee (label: none), however "
            "large or unfavorable."
        ),
        "example_cue_words": ["may be chargeable", "nominal fee", "as applicable", "subject to change"],
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
        "label": "Obscure / discretionary penalty",
        "description": (
            "Penalty, late-fee, or default terms where the ACTUAL AMOUNT "
            "OR TRIGGER IS NOT FULLY SPECIFIED in the clause -- vague "
            "('such other amount as may be determined'), open-ended "
            "escalation, or genuinely confusing/compounding calculation. "
            "A late-fee table with concrete numbers, or a stated maximum "
            "percentage, is NOT this category (label: none) even though "
            "it is a real cost to the consumer."
        ),
        "example_cue_words": ["as may be determined", "at its discretion", "up to a maximum"],
    },
    "deceptive_framing": {
        "label": "Unconstrained discretionary power",
        "description": (
            "The company holds a BROAD, VAGUELY-BOUNDED power over "
            "something consequential to the consumer -- rejecting, "
            "modifying, cancelling, or suspending an account, facility, "
            "or benefit -- exercised 'at its sole discretion', 'without "
            "assigning any reason', or 'without notice'. The power's "
            "EXISTENCE being disclosed does not exempt it from this "
            "category; what matters is that its exercise is effectively "
            "unconstrained from the consumer's side."
        ),
        "example_cue_words": ["at its sole discretion", "without assigning any reason", "without prior notice"],
    },
    "forced_arbitration": {
        "label": "Forced arbitration / dispute-limiting clause",
        "description": (
            "Actual arbitration clauses, OR language stating a decision "
            "'shall be final and binding' on the consumer, OR clauses "
            "that waive/limit the consumer's ability to dispute or seek "
            "recourse, OR unusually short mandatory claim-filing windows. "
            "A simple payment demand, legal-action warning, or credit-"
            "bureau reporting clause is NOT this category -- those are "
            "ordinary consequences of default, not dispute-limiting "
            "language, and belong under penalty_obscurity (if the amount "
            "is unclear) or none (if it is clear)."
        ),
        "example_cue_words": ["final and binding", "arbitration", "waive", "sole arbitrator"],
    },
    "none": {
        "label": "No dark pattern detected",
        "description": (
            "A neutral clause, OR a clearly quantified fee/penalty/term, "
            "OR a standard liability/insurance disclaimer, OR an ordinary "
            "consequence of default (credit bureau reporting, legal "
            "action) with no vagueness about amount or trigger."
        ),
        "example_cue_words": [],
    },
}

CATEGORY_IDS = list(CATEGORIES.keys())
