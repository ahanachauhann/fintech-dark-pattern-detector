"""
Splits raw T&C .txt files (in data/raw/) into clause-level rows and
writes them to data/labeled/clauses_unlabeled.csv, ready for labeling.

Usage:
    python src/clause_split.py
"""
import csv
import re
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "labeled" / "clauses_unlabeled.csv"


# Common abbreviations that end in a period but do NOT end a sentence.
# Without this, "Rs. 10,000" or "etc. For example" gets wrongly split in two.
_ABBREVIATIONS = {
    "rs", "mr", "mrs", "dr", "vs", "etc", "no", "ltd", "pvt", "co",
    "govt", "regd", "acc", "inc", "corp", "jr", "sr", "u.s", "i.e", "e.g",
}


def _merge_false_splits(parts: list[str]) -> list[str]:
    """Re-join fragments that were split after an abbreviation, or where
    the next fragment starts with a digit (a continuation of a number,
    amount, or list item rather than a new sentence)."""
    merged: list[str] = []
    for part in parts:
        if merged:
            prev = merged[-1]
            last_word = re.findall(r"(\w+)\.$", prev.strip())
            ends_in_abbrev = bool(last_word) and last_word[0].lower() in _ABBREVIATIONS
            next_starts_with_digit = bool(re.match(r"^\d", part.strip()))
            if ends_in_abbrev or next_starts_with_digit:
                merged[-1] = prev.rstrip() + " " + part.lstrip()
                continue
        merged.append(part)
    return merged


def split_into_clauses(text: str) -> list[str]:
    """Split text into clause-level chunks (roughly one sentence/bullet each)."""
    # Split on newlines first (T&C docs are often already bullet/line separated)
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    clauses = []
    for line in lines:
        # Further split long lines into sentences, then repair false splits
        # caused by abbreviations (e.g. "Rs.") or numbers (e.g. "Rs. 1,000").
        raw_sentences = re.split(r"(?<=[.;])\s+", line)
        sentences = _merge_false_splits(raw_sentences)
        for s in sentences:
            s = s.strip()
            if len(s.split()) >= 5:  # skip very short fragments/headers
                clauses.append(s)
    return clauses


def infer_product_category(filename: str) -> tuple[str, str]:
    """filename convention: <category>_<product>.txt"""
    stem = filename.rsplit(".", 1)[0]
    parts = stem.split("_", 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return "unknown", stem


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for txt_file in sorted(RAW_DIR.glob("*.txt")):
        category, product = infer_product_category(txt_file.name)
        text = txt_file.read_text(encoding="utf-8")
        for clause in split_into_clauses(text):
            rows.append({
                "product": product,
                "category": category,
                "source_file": txt_file.name,
                "clause_text": clause,
                "label": "",  # to be filled by label_with_llm.py or manually
            })

    with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["product", "category", "source_file", "clause_text", "label"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} clauses from {len(list(RAW_DIR.glob('*.txt')))} files to {OUT_PATH}")


if __name__ == "__main__":
    main()
