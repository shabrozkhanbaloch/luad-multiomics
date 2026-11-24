#!/usr/bin/env python
"""Step 1. Nested cross-validation: every arm, every scenario, both models.

    python scripts/01_cv.py                 # the full run
    python scripts/01_cv.py --quick         # one repeat, tiny grids: a smoke test
    python scripts/01_cv.py --endpoint os   # the sensitivity endpoint
    python scripts/01_cv.py --permute       # shuffled outcome: everything should be 0.50

Writes results/tables/cv_<endpoint>[_permuted]_predictions.parquet (one risk
per fold, model, arm and test patient), cv_..._info.csv (chosen widths and
selected features) and cv_..._summary.csv (mean concordance per arm).
Step 2 turns the predictions into intervals.
"""
from __future__ import annotations

import argparse
import time

from _common import load_config, n_jobs, out_dirs, save_predictions

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from luad import models as M
from luad.cv import outer_folds, run_fold
from luad.data import load_cohort
from pathlib import Path


def summarise(results: list[dict], y: np.ndarray) -> pd.DataFrame:
    rows = []
    for r in results:
        for model, arms in r["risk"].items():
            for arm, risk in arms.items():
                rows.append({"model": model, "arm": arm,
                             "uno_c": M.uno_c(y[r["train"]], y[r["test"]], risk),
                             "harrell_c": M.harrell_c(y[r["test"]], risk)})
    df = pd.DataFrame(rows)
    return (df.groupby(["model", "arm"], sort=False)
              .agg(uno_c=("uno_c", "mean"), harrell_c=("harrell_c", "mean"),
                   folds=("uno_c", "size")).reset_index())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--endpoint", default=None)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--permute", action="store_true")
    ap.add_argument("--arms", nargs="+", default=None, help="subset of arms")
    ap.add_argument("--jobs", type=int, default=None)
    ap.add_argument("--tag", default=None, help="suffix for the output files")
    args = ap.parse_args()

    cfg = load_config()
    seed = int(cfg["project"]["seed"])
    endpoint = (args.endpoint or cfg["endpoint"]["primary"]).lower()
    tables, _ = out_dirs(cfg)
    stem = f"cv_{endpoint}" + ("_permuted" if args.permute else "") \
        + (f"_{args.tag}" if args.tag else "") + ("_quick" if args.quick else "")

    print(f"NESTED CROSS-VALIDATION  ({stem})")
    c = load_cohort(cfg, endpoint)
    if args.permute:
        # (event, time) stay paired, so the event rate and censoring are
        # unchanged; only the link from features to outcome is destroyed.
        c = c.with_outcome(c.y[np.random.default_rng(seed).permutation(c.n)])
        print("  outcome permuted: every concordance below should sit at 0.50")

    folds = outer_folds(c, cfg, seed, args.quick)
    cache = Path(cfg["paths"]["cache"]) / "mofa"
    t0 = time.time()
    results = Parallel(n_jobs=min(n_jobs(args.jobs), len(folds)), verbose=5)(
        delayed(run_fold)(c, tr, te, fid, cfg, seed, args.quick, 1, cache, args.arms)
        for fid, tr, te in folds)

    save_predictions(results, tables / f"{stem}_predictions.parquet")
    pd.DataFrame([row for r in results for row in r["info"]]).to_csv(
        tables / f"{stem}_info.csv", index=False)
    summary = summarise(results, c.y)
    summary.to_csv(tables / f"{stem}_summary.csv", index=False)
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\n{len(folds)} folds in {(time.time() - t0) / 60:.1f} min -> {tables}/{stem}_*")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
