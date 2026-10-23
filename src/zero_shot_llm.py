"""
Zero-shot LLM classification: the third leg of the comparison.

The model is given ONLY the category definitions from categories.py (no labeled
examples, no training) and asked to pick one category per clause. Because nothing
is trained, every clause is predicted once; per-fold scores are then computed on
the SAME test folds as TF-IDF and DistilBERT, so compare_cv.py works unchanged.

Backends:
  hf         open-weights model on a GPU (e.g. Kaggle). No API key needed.
  anthropic  Claude via API. Needs ANTHROPIC_API_KEY in the environment.
  dummy      always answers "none" (pipeline test only; never report this).

Usage:
    python src/zero_shot_llm.py --backend hf --model Qwen/Qwen2.5-7B-Instruct --name zeroshot_qwen7b
    python src/zero_shot_llm.py --backend anthropic --model claude-haiku-5-5 --name zeroshot_claude
Then:
    python src/compare_cv.py tfidf zeroshot_qwen7b grouped

Predictions are cached in results/<name>_predictions.csv, so an interrupted run resumes.
Honest-reporting notes: unparseable answers are counted and scored as "none"; the
category definitions were written AFTER looking at the data, so this is zero-shot on
the labeling, not on the taxonomy's design (state this in the report).
"""
import argparse
import csv
import json
import time
from collections import defaultdict

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from categories import CATEGORIES, CATEGORY_IDS
from cv_splits import ROOT, SPLITS_PATH, TAG, load_rows

SYSTEM = (
    "You are an expert analyst of consumer financial contracts. You classify one clause "
    "from a loan, buy-now-pay-later, credit card or insurance terms-and-conditions "
    "document into exactly one category. Reply with the category id only."
)


GUIDE_V3 = """Categories (decide in this order and stop at the first that fits):
1. deceptive_framing: the company gives itself broad unilateral power - 'sole discretion', 'without notice or reason', changing terms or services at any time, rejecting, suspending, cancelling or disclosing at will. If the discretion is over a charge, it is still deceptive_framing.
2. penalty_obscurity: ONLY a late/default money penalty whose amount or trigger is unspecified or escalating. The word 'penalty' alone does not count.
3. hidden_fee: a fee, charge or interest component whose amount, rate or trigger/threshold is not stated; fees that 'may vary'; charges levied outside the headline rate. A fully stated amount is none.
4. forced_arbitration: an arbitration clause, a waiver of the right to dispute or sue, or a deemed-acceptance window (e.g. statement deemed correct unless disputed within N days). A bank's own 'final and binding decision' is deceptive_framing, not this.
5. none: rules that protect or inform the customer (disclosure requirements, notice promises); fully stated terms; plain information; standard 'not liable for events beyond our control'; ordinary default consequences such as credit-bureau reporting or legal action; reward points lapsing."""
V3_IDS = ["deceptive_framing", "penalty_obscurity", "hidden_fee", "forced_arbitration", "none"]


def build_prompt(clause, with_cues=False, prompt_version="v2"):
    if prompt_version == "v3":
        return (GUIDE_V3 + f"\n\nClause:\n\"\"\"{clause}\"\"\"\n\nAnswer with exactly one of: {', '.join(V3_IDS)}")
    lines = ["Categories:"]
    for cid in CATEGORY_IDS:
        c = CATEGORIES[cid]
        line = f"- {cid}: {c['label']}. {c['description']}"
        if with_cues and c.get("example_cue_words"):
            line += f" Cue phrases: {', '.join(c['example_cue_words'])}."
        lines.append(line)
    lines.append("\nRule of thumb: a clause is a dark pattern because of HOW it is written "
                 "(vague, unquantified, discretionary, or disclosed after the fact), not merely "
                 "because it mentions money or a company right.")
    lines.append(f"\nClause:\n\"\"\"{clause}\"\"\"")
    lines.append(f"\nAnswer with exactly one of: {', '.join(CATEGORY_IDS)}")
    return "\n".join(lines)


def parse_label(text):
    t = text.lower()
    hits = [(t.find(cid), cid) for cid in CATEGORY_IDS if cid in t]
    return min(hits)[1] if hits else None


def make_backend(args):
    if args.backend == "dummy":
        return lambda system, user: "none"
    if args.backend == "anthropic":
        import anthropic
        client = anthropic.Anthropic()

        def run(system, user):
            for attempt in range(5):
                try:
                    try:
                        r = client.messages.create(model=args.model, max_tokens=64, temperature=0,
                                                   system=system, messages=[{"role": "user", "content": user}])
                    except anthropic.BadRequestError:  # some models reject temperature
                        r = client.messages.create(model=args.model, max_tokens=64,
                                                   system=system, messages=[{"role": "user", "content": user}])
                    # join text blocks only (a reasoning block, if any, has no .text answer)
                    return "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
                except (anthropic.RateLimitError, anthropic.APIConnectionError, anthropic.InternalServerError):
                    time.sleep(2 ** attempt)
            raise RuntimeError("API failed repeatedly")
        return run
    if args.backend == "hf":
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.model)
        extra = {}
        if args.load_in_4bit:
            from transformers import BitsAndBytesConfig
            extra["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        try:
            model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float16, device_map="auto", **extra)
        except TypeError:
            model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16, device_map="auto", **extra)
        model.eval()

        def run(system, user):
            msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            ids = tok(text, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(**ids, max_new_tokens=16, do_sample=False, pad_token_id=tok.eos_token_id)
            return tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)
        return run
    raise ValueError(args.backend)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["hf", "anthropic", "dummy"], required=True)
    ap.add_argument("--model", default="")
    ap.add_argument("--name", default="zeroshot", help="method name used in results file names")
    ap.add_argument("--with-cues", action="store_true", help="also show cue phrases (off by default: they mirror the labeling rules)")
    ap.add_argument("--prompt", choices=["v2", "v3"], default="v2",
                    help="v2 = original definitions (used in the first run); v3 = refined annotation guideline")
    ap.add_argument("--load-in-4bit", action="store_true", help="hf backend: 4-bit quantization for larger models (needs bitsandbytes)")
    ap.add_argument("--limit", type=int, default=None, help="smoke test: only first N clauses (skips CV scoring)")
    args = ap.parse_args()

    rows = load_rows()
    n = args.limit or len(rows)
    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    pred_path = out_dir / f"{args.name}_predictions.csv"

    cache = {}
    if pred_path.exists():
        with open(pred_path, newline="", encoding="utf-8") as f:
            cache = {int(r["idx"]): r for r in csv.DictReader(f)}

    run = make_backend(args) if any(i not in cache for i in range(n)) else None   # no model load if fully cached
    new = not pred_path.exists()
    with open(pred_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["idx", "true", "pred", "raw"])
        if new:
            w.writeheader()
        for i in range(n):
            if i in cache:
                continue
            raw = run(SYSTEM, build_prompt(rows[i]["clause_text"], args.with_cues, args.prompt)).strip()
            pred = parse_label(raw)
            rec = {"idx": i, "true": rows[i]["label"], "pred": pred or "UNPARSED", "raw": raw.replace("\n", " ")[:80]}
            w.writerow(rec)
            f.flush()
            cache[i] = rec
            if (i + 1) % 25 == 0:
                print(f"{i+1}/{n} done", flush=True)

    if args.limit:
        print(f"Smoke test done ({n} clauses). Sample outputs:")
        for i in range(min(n, 8)):
            print(f"  true={cache[i]['true']:20s} pred={cache[i]['pred']:20s} raw={cache[i]['raw']!r}")
        return

    y = np.array([rows[i]["label"] for i in range(len(rows))])   # current label version, not the cached one
    raw_pred = np.array([cache[i]["pred"] for i in range(len(rows))])
    n_unparsed = int((raw_pred == "UNPARSED").sum())
    pred = np.where(raw_pred == "UNPARSED", "none", raw_pred)
    labels = sorted(set(y))
    off_taxonomy = {c: int((pred == c).sum()) for c in CATEGORY_IDS if c not in labels and (pred == c).any()}

    splits_all = json.loads(SPLITS_PATH.read_text())
    for mode in ("stratified", "grouped"):
        accs, macros = [], []
        per_class = defaultdict(list)
        for _, te in splits_all[mode]:
            te = np.array(te)
            accs.append(accuracy_score(y[te], pred[te]))
            macros.append(f1_score(y[te], pred[te], labels=sorted(set(y[te])), average="macro", zero_division=0))
            for lab, fv in zip(labels, f1_score(y[te], pred[te], labels=labels, average=None, zero_division=0)):
                if (y[te] == lab).any():
                    per_class[lab].append(fv)
        lines = [f"{args.name} ({args.backend} {args.model}, prompt {args.prompt}) | mode={mode} | {len(rows)} clauses | {len(accs)} folds | zero-shot",
                 f"accuracy : {np.mean(accs):.3f} +/- {np.std(accs):.3f}",
                 f"macro-F1 : {np.mean(macros):.3f} +/- {np.std(macros):.3f}",
                 f"unparseable answers (scored as none): {n_unparsed}; predicted classes absent from data: {off_taxonomy or 'none'}",
                 "per-class F1 (mean +/- std over folds where the class appears):"]
        for lab in labels:
            v = per_class[lab]
            lines.append(f"  {lab:20s} {np.mean(v):.3f} +/- {np.std(v):.3f}  (n_folds={len(v)})")
        report = "\n".join(lines)
        print(report + "\n")
        (out_dir / f"{args.name}_cv_{mode}{TAG}.txt").write_text(report + "\n")
        json.dump({"accuracy": accs, "macro_f1": macros}, open(out_dir / f"{args.name}_cv_{mode}{TAG}.json", "w"))


if __name__ == "__main__":
    main()
