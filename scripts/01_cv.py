#!/usr/bin/env python
"""Step 1. Nested cross-validation: every arm, every scenario, both models.

    python scripts/01_cv.py                 # the full run
    python scripts/01_cv.py --quick         # one repeat, tiny grids: a smoke test
    python scripts/01_cv.py --endpoint os   # the sensitivity endpoint
    python scripts/01_cv.py --permute       # shuffled outcome: everything should be 0.50

What it is for
    This is the step that fits the models. The patients are split into five
    parts (folds). Each part in turn is held back as the test patients, the
    models are trained on the rest, and a risk is predicted for every test
    patient. This is repeated five times with different splits, which gives
    25 folds. Every arm (a way of using the data blocks), every scenario
    (what the test patient is missing) and both models (penalised Cox, random
    survival forest) are fitted on the same folds, so they can be compared
    patient by patient. "Nested" means the model settings are chosen in a
    second cross-validation inside the training patients only.

What it reads
    config/config.yaml      the plan: blocks, models, folds, scenarios
    data/processed/         the four data blocks
    data/interim/           the endpoints (event and time per patient)

What it writes
    In results/tables/, where <name> is cv_<endpoint>, followed by _permuted
    with --permute, by _<tag> with --tag and by _quick with --quick:

    <name>_predictions.parquet   one risk per fold, model, arm and test patient
    <name>_info.csv              chosen widths and selected features
    <name>_summary.csv           mean concordance per arm

    Step 2 turns the predictions into intervals.

How long it takes
    About 37 minutes on a 12-worker machine (35 minutes for --endpoint os,
    10 minutes for --permute --arms clinical early mofa). With --quick,
    30 seconds.
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
    """Mean concordance per model and arm, over the folds.

    Arguments
        results   one dict per fold, as returned by luad.cv.run_fold:
                  {"train": row numbers, "test": row numbers,
                   "risk": {model: {arm: one risk per test patient}}, ...}
        y         the outcome of every patient in the cohort (event and time)

    Returns
        One row per model and arm, with the columns model, arm, uno_c,
        harrell_c (both are means over folds) and folds (how many folds).
    """
    rows = []
    for fold_result in results:
        for model, arms in fold_result["risk"].items():
            for arm, risk in arms.items():
                # Uno's C also needs the training outcomes: it weights the test
                # patients by the censoring pattern seen in training. Harrell's C
                # uses the test patients alone.
                rows.append({"model": model, "arm": arm,
                             "uno_c": M.uno_c(y[fold_result["train"]], y[fold_result["test"]], risk),
                             "harrell_c": M.harrell_c(y[fold_result["test"]], risk)})
    per_fold = pd.DataFrame(rows)
    return (per_fold.groupby(["model", "arm"], sort=False)
                    .agg(uno_c=("uno_c", "mean"), harrell_c=("harrell_c", "mean"),
                         folds=("uno_c", "size")).reset_index())


def main() -> int:
    """Run the cross-validation and write its three tables.

    Returns
        0, the exit code for success.
    """
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--endpoint", default=None,
                        help="endpoint to analyse, pfi or os (default: the primary one in the config)")
    parser.add_argument("--quick", action="store_true",
                        help="one repeat and tiny grids: a smoke test, not a result")
    parser.add_argument("--permute", action="store_true",
                        help="shuffle the outcome first; every concordance should then be 0.50")
    parser.add_argument("--arms", nargs="+", default=None, help="subset of arms")
    parser.add_argument("--jobs", type=int, default=None,
                        help="number of worker processes (default: most of the CPU threads)")
    parser.add_argument("--tag", default=None, help="suffix for the output files")
    args = parser.parse_args()

    # ---- Read the plan ----
    cfg = load_config()
    seed = int(cfg["project"]["seed"])
    endpoint = (args.endpoint or cfg["endpoint"]["primary"]).lower()
    tables, _ = out_dirs(cfg)
    # The file name records how the run was made, so a quick, permuted or
    # tagged run can never overwrite the tables of the full run.
    stem = f"cv_{endpoint}" + ("_permuted" if args.permute else "") \
        + (f"_{args.tag}" if args.tag else "") + ("_quick" if args.quick else "")

    # ---- Load the data ----
    print(f"NESTED CROSS-VALIDATION  ({stem})")
    cohort = load_cohort(cfg, endpoint)
    if args.permute:
        # (event, time) stay paired, so the event rate and censoring are
        # unchanged; only the link from features to outcome is destroyed.
        cohort = cohort.with_outcome(cohort.y[np.random.default_rng(seed).permutation(cohort.n)])
        print("  outcome permuted: every concordance below should sit at 0.50")

    # ---- Compute: one job per outer fold ----
    # The folds are fixed here, once, from the seed. Every arm and both models
    # then see exactly the same training and test patients.
    folds = outer_folds(cohort, cfg, seed, args.quick)
    # MOFA is the slowest fit. Its result is kept in a cache folder so a rerun
    # of the same fold does not fit it again.
    cache = Path(cfg["paths"]["cache"]) / "mofa"
    start_time = time.time()
    # The folds do not depend on each other, so they run side by side, one
    # worker per fold. Each worker uses a single thread (the "1" below).
    results = Parallel(n_jobs=min(n_jobs(args.jobs), len(folds)), verbose=5)(
        delayed(run_fold)(cohort, train_rows, test_rows, fold_id, cfg, seed, args.quick, 1, cache,
                          args.arms)
        for fold_id, train_rows, test_rows in folds)

    # ---- Write the tables ----
    save_predictions(results, tables / f"{stem}_predictions.parquet")
    # Each fold reports one row per model and arm (chosen width, selected
    # features); the rows of all folds are put into one table.
    pd.DataFrame([row for fold_result in results for row in fold_result["info"]]).to_csv(
        tables / f"{stem}_info.csv", index=False)
    summary = summarise(results, cohort.y)
    summary.to_csv(tables / f"{stem}_summary.csv", index=False)

    # ---- Print the summary ----
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\n{len(folds)} folds in {(time.time() - start_time) / 60:.1f} min -> {tables}/{stem}_*")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
