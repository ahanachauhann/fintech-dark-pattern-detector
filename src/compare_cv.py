"""
Paired comparison of two methods scored on the SAME cross-validation folds.

Usage:
    python src/compare_cv.py tfidf distilbert stratified
    python src/compare_cv.py tfidf distilbert grouped

Reads results/<name>_cv_<mode>.json (per-fold scores written by the eval scripts).
Reports the mean paired difference, how many folds each method wins, and a
Nadeau-Bengio corrected resampled t-test. Plain t-tests on repeated-CV folds are
too optimistic because training sets overlap; the correction inflates the variance.
"""
import json
import sys

import numpy as np
from scipy import stats

from cv_splits import ROOT

TEST_OVER_TRAIN = 0.25  # 5-fold: test = 20%, train = 80%


def load(name, mode):
    return json.load(open(ROOT / "results" / f"{name}_cv_{mode}.json"))


def main():
    a_name, b_name, mode = sys.argv[1:4]
    A, B = load(a_name, mode), load(b_name, mode)
    for metric in ("macro_f1", "accuracy"):
        a, b = np.array(A[metric]), np.array(B[metric])
        d = b - a                                  # positive = B better
        k = len(d)
        var = d.var(ddof=1)
        t = d.mean() / np.sqrt((1 / k + TEST_OVER_TRAIN) * var)
        p = 2 * stats.t.sf(abs(t), df=k - 1)
        print(f"[{mode}] {metric:9s} {a_name}={a.mean():.3f}  {b_name}={b.mean():.3f}  "
              f"diff({b_name}-{a_name})={d.mean():+.3f}  {b_name} wins {int((d>0).sum())}/{k} folds, "
              f"ties {int((d==0).sum())}  corrected-t p={p:.3f}")


if __name__ == "__main__":
    main()
