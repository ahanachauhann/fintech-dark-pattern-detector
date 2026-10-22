"""
Builds the corrected dataset (v3) from the original labels (v2) plus the hand review.

Inputs (all in the repo):
  data/labeled/clauses_reviewed.csv      original labels (v2, tagged labels-v2-unverified)
  review/review_sheet.csv + review_key.csv   40-clause blind review (your labels vs v2); clause #35 keeps v2 label
  review/relabel_queue.csv + relabel_labels.json   76-clause targeted relabel (qid -> your label)

Outputs:
  data/labeled/clauses_reviewed_v3.csv   same columns, corrected labels
  results/relabel_changelog.csv          every changed row: v2 label, v3 label, source
v2 files are never modified.
"""
import csv
import json

from cv_splits import ROOT

KEEP_V2_IN_FIRST_REVIEW = {35}   # decided: clause #35 keeps its original label


def main():
    src = ROOT / "data" / "labeled" / "clauses_reviewed.csv"
    with open(src, newline="", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        fields = rd.fieldnames
        rows = [r for r in rd]
    v2 = [r["label"] for r in rows]
    new = {}      # row_index -> (label, source)

    sheet = {int(r["id"]): r["my_label"].strip() for r in csv.DictReader(open(ROOT / "review/review_sheet.csv", encoding="utf-8"))}
    for k in csv.DictReader(open(ROOT / "review/review_key.csv", encoding="utf-8")):
        i, j = int(k["id"]), int(k["row_index"])
        if i in KEEP_V2_IN_FIRST_REVIEW or not sheet.get(i):
            continue
        new[j] = (sheet[i], "review40")

    labels = {int(k): v for k, v in json.load(open(ROOT / "review/relabel_labels.json")).items()}
    queue = {int(r["qid"]): int(r["row_index"]) for r in csv.DictReader(open(ROOT / "review/relabel_queue.csv", encoding="utf-8"))}
    missing = [q for q in queue if q not in labels]
    for q, lab in labels.items():
        new[queue[q]] = (lab, "relabel76")

    changed = []
    for j, (lab, srcname) in new.items():
        rows[j]["label"] = lab
        if lab != v2[j]:
            changed.append((j, v2[j], lab, srcname, rows[j]["clause_text"]))
    out = ROOT / "data" / "labeled" / "clauses_reviewed_v3.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    (ROOT / "results").mkdir(exist_ok=True)
    with open(ROOT / "results" / "relabel_changelog.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["row_index", "v2_label", "v3_label", "source", "clause_text"])
        w.writerows(sorted(changed))
    print(f"reviewed rows: {len(new)} | labels changed: {len(changed)} | relabel queue not yet labeled: {len(missing)}")


if __name__ == "__main__":
    main()
