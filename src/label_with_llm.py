"""
First-pass labeling of clauses using an LLM API (Anthropic Claude).

This gives you a fast starting point — you MUST manually review and
correct a meaningful sample afterward. Document this process clearly
in your report's methodology section (this is a legitimate, common
technique, not something to hide).

Setup:
    pip install anthropic
    export ANTHROPIC_API_KEY=your_key_here

Usage:
    python src/label_with_llm.py
"""
import csv
import os
import time
from pathlib import Path

from categories import CATEGORIES, CATEGORY_IDS

IN_PATH = Path(__file__).resolve().parent.parent / "data" / "labeled" / "clauses_unlabeled.csv"
OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "labeled" / "clauses_llm_labeled.csv"

CATEGORY_PROMPT_BLOCK = "\n".join(
    f"- {cid}: {info['label']} — {info['description']}" for cid, info in CATEGORIES.items()
)


def build_prompt(clause_text: str) -> str:
    return f"""You are labeling a single clause from a financial product's terms & conditions for a research project on manipulative "dark patterns" in fintech contracts.

Categories:
{CATEGORY_PROMPT_BLOCK}

Clause:
\"\"\"{clause_text}\"\"\"

Reply with ONLY the category id (one of: {", ".join(CATEGORY_IDS)}) that best fits this clause. If it's a neutral, clearly disclosed clause with no manipulative pattern, reply "none"."""


def label_clause(client, clause_text: str) -> str:
    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=20,
        messages=[{"role": "user", "content": build_prompt(clause_text)}],
    )
    reply = message.content[0].text.strip().lower()
    return reply if reply in CATEGORY_IDS else "none"


def main():
    try:
        import anthropic
    except ImportError:
        raise SystemExit("Run: pip install anthropic")

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise SystemExit("Set ANTHROPIC_API_KEY environment variable first.")

    client = anthropic.Anthropic(api_key=api_key)

    with open(IN_PATH, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    for i, row in enumerate(rows):
        row["label"] = label_clause(client, row["clause_text"])
        if i % 10 == 0:
            print(f"Labeled {i}/{len(rows)}")
        time.sleep(0.2)  # be gentle on rate limits

    with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["product", "category", "source_file", "clause_text", "label"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote LLM first-pass labels to {OUT_PATH}")
    print("NEXT STEP: open this CSV and manually review/correct a sample of the labels.")


if __name__ == "__main__":
    main()
