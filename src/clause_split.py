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


def split_into_clauses(text: str) -> list[str]:
    """Split text into clause-level chunks (roughly one sentence/bullet each)."""
    # Split on newlines first (T&C docs are often already bullet/line separated)
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    clauses = []
    for line in lines:
        # Further split long lines into sentences
        sentences = re.split(r"(?<=[.;])\s+", line)
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
