#!/usr/bin/env python
"""Step 3. External validation: models trained on TCGA, tested on GEO.

    python scripts/03_external.py
    python scripts/03_external.py --quick

What it is for
    Cross-validation tests a model on patients from the same study it was
    trained on. This step asks the harder question: do the models still work
    on patients from other hospitals and other measuring platforms? Each
    model is trained on all TCGA patients and applied unchanged to three GEO
    cohorts. Each GEO patient has expression and clinical data only, so
    mutation and methylation are missing for real, not by simulation. The
    arms that have to cope with that are written with `@geo`.

What it reads
    config/config.yaml                       the cohorts, endpoints and contrasts (under `external`)
    data/processed/, data/interim/           the TCGA blocks and endpoints
    data/processed/geo_<cohort>_*.parquet    expression and clinical data of each GEO cohort

What it writes
    results/tables/external_arms.csv         concordance of each arm in each cohort, with its interval
    results/tables/external_contrasts.csv    each contrast in each cohort: difference, interval, verdict
    results/figures/external.png             one panel per cohort and endpoint

    With --quick the three names end in _quick, so the full results are kept.

How long it takes
    About 15 minutes on a 12-worker machine.
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path

from _common import load_config, n_jobs, out_dirs

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from joblib import Parallel, delayed

from luad.bootstrap import paired_bootstrap
from luad.cv import run_fold
from luad.data import load_cohort
from luad.external import load_geo, stack

# The arms that are fitted: the ones that can be scored on a patient who has
# clinical and expression data only.
ARMS = ["clinical", "clinical+rna", "early", "late", "mofa"]
# The results that are reported. The integration arms are taken from the
# `geo` scenario (mutation and methylation missing), not from the full one.
KEEP = ["clinical", "clinical+rna", "early@geo", "late@geo", "mofa@geo"]


def one(cfg, tcga_endpoint, geo_endpoint, cohort_id, quick, n_boot):
    """Train on TCGA, test on one GEO cohort for one endpoint, and bootstrap the result.

    Arguments
        cfg             the plan, as returned by load_config()
        tcga_endpoint   the endpoint the models are trained on, "pfi" or "os"
        geo_endpoint    the matching endpoint in the GEO cohort, "rfs" or "os"
        cohort_id       the GEO series, for example "GSE72094"
        quick           True for tiny grids (a smoke test)
        n_boot          number of bootstrap resamples

    Returns
        (arms table, contrasts table) for this cohort and endpoint, with both
        models one under the other.
    """
    # The plan is copied before it is changed, so the change stays inside this
    # job. The only scenarios needed here are "nothing missing" and "what a
    # GEO patient is missing".
    cfg = copy.deepcopy(cfg)
    cfg["scenarios"] = {"full": [], "geo": ["mutation", "methylation"]}
    seed = int(cfg["project"]["seed"])

    # TCGA and GEO patients are put into one table: the TCGA rows are the
    # training patients and the GEO rows the test patients. This makes
    # external validation one more "fold" for the same code as step 1.
    tcga = load_cohort(cfg, tcga_endpoint, verbose=False)
    geo_expression, geo_clinical, y_geo = load_geo(cfg, cohort_id, geo_endpoint)
    both, train_rows, test_rows = stack(tcga, geo_expression, geo_clinical, y_geo,
                                        cfg["external"]["shared_clinical"])
    fold_result = run_fold(both, train_rows, test_rows, f"{cohort_id}_{geo_endpoint}", cfg, seed,
                           quick, 1, Path(cfg["paths"]["cache"]) / "mofa", ARMS)

    arms_all, contrasts_all = [], []
    for model, risks in fold_result["risk"].items():
        # Uno's weights need the censoring distribution of the cohort being
        # scored, so the reference rows are the GEO cohort itself.
        fold = {"train": test_rows, "test": test_rows,
                "risk": {arm: risk for arm, risk in risks.items() if arm in KEEP}}
        bootstrap_settings = cfg["bootstrap"]
        arms, contrasts = paired_bootstrap([fold], both.y,
                                           [tuple(pair) for pair in cfg["external"]["contrasts"]],
                                           n_boot, seed,
                                           int(bootstrap_settings["min_events_per_resample"]),
                                           float(bootstrap_settings["level"]))
        # Label every row with where it comes from. Each column is put in
        # front, so the last one inserted (cohort) ends up first.
        for table in (arms, contrasts):
            table.insert(0, "model", model)
            table.insert(0, "n_events", int(y_geo["event"].sum()))
            table.insert(0, "n", len(test_rows))
            table.insert(0, "endpoint", f"{tcga_endpoint.upper()}->{geo_endpoint.upper()}")
            table.insert(0, "cohort", cohort_id)
        arms_all.append(arms)
        contrasts_all.append(contrasts)
    return pd.concat(arms_all), pd.concat(contrasts_all)


def figure(arms: pd.DataFrame, path) -> None:
    """Draw one panel per cohort and endpoint: Uno's C of each arm with its interval.

    Arguments
        arms   the arms table of every cohort (as written to external_arms.csv)
        path   the image file to write

    Returns
        Nothing. The figure is saved to `path`.
    """
    uno_rows = arms[arms["metric"] == "uno_c"]
    panels = list(uno_rows.groupby(["cohort", "endpoint"], sort=False))
    fig, axes = plt.subplots(1, len(panels), figsize=(3.6 * len(panels), 3.6), sharey=True)
    for ax, ((cohort_id, endpoint_label), panel_rows) in zip(np_atleast(axes), panels):
        for model_number, (model, model_rows) in enumerate(panel_rows.groupby("model", sort=False)):
            model_rows = model_rows.set_index("arm").loc[KEEP]
            # The two models are drawn a little above and below each arm's
            # line, so their intervals do not cover each other.
            heights = [i + (model_number - 0.5) * 0.3 for i in range(len(KEEP))]
            ax.errorbar(model_rows["c_index"], heights,
                        xerr=[model_rows["c_index"] - model_rows["ci_low"],
                              model_rows["ci_high"] - model_rows["c_index"]],
                        fmt="o", ms=4, lw=1.2, label=model,
                        color=["#2a78d6", "#c2541c"][model_number])
        # The dashed line marks 0.5, the concordance of a random guess.
        ax.axvline(0.5, color="#898781", lw=0.8, ls="--")
        ax.set_yticks(range(len(KEEP)))
        ax.set_yticklabels(KEEP, fontsize=8)
        ax.set_title(f"{cohort_id}\n{endpoint_label}  (n={panel_rows['n'].iloc[0]}, "
                     f"{panel_rows['n_events'].iloc[0]} events)", fontsize=9)
        ax.set_xlabel("Uno's C")
    np_atleast(axes)[0].legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def np_atleast(axes):
    """The panels of a figure as a list-like array, also when there is only one.

    Arguments
        axes   what plt.subplots returned: one panel, or an array of panels

    Returns
        An array of panels, so the caller can always loop over it.
    """
    import numpy as np
    return np.atleast_1d(axes)


def main() -> int:
    """Run every cohort and endpoint, then write the two tables and the figure.

    Returns
        0, the exit code for success.
    """
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true",
                        help="tiny grids and 200 resamples; the output names end in _quick")
    parser.add_argument("--jobs", type=int, default=None,
                        help="number of worker processes (default: most of the CPU threads)")
    args = parser.parse_args()

    # ---- Read the plan ----
    cfg = load_config()
    tables, figures = out_dirs(cfg)
    n_boot = 200 if args.quick else int(cfg["bootstrap"]["n_resamples"])

    # ---- Load the data: decide which cohort and endpoint pairs can be run ----
    # Not every GEO cohort records every endpoint. A pair is run only if the
    # cohort has at least 20 events for it; this is how a cohort without the
    # endpoint is recognised.
    jobs = []
    for tcga_endpoint, geo_endpoint in cfg["external"]["endpoints"].items():
        for cohort_id in cfg["external"]["cohorts"]:
            geo_clinical = pd.read_parquet(
                Path(cfg["paths"]["processed"]) / f"geo_{cohort_id}_clinical.parquet")
            if geo_clinical[f"{geo_endpoint}_event"].sum() >= 20:        # the cohort records this endpoint
                jobs.append((tcga_endpoint.lower(), geo_endpoint, cohort_id))
            else:
                print(f"  {cohort_id}: no usable {geo_endpoint.upper()} endpoint, skipped")

    # ---- Compute: one job per cohort and endpoint, side by side ----
    results = Parallel(n_jobs=min(n_jobs(args.jobs), len(jobs)), verbose=5)(
        delayed(one)(cfg, tcga_endpoint, geo_endpoint, cohort_id, args.quick, n_boot)
        for tcga_endpoint, geo_endpoint, cohort_id in jobs)

    # ---- Write the tables ----
    arms = pd.concat([job_arms for job_arms, _ in results], ignore_index=True)
    contrasts = pd.concat([job_contrasts for _, job_contrasts in results], ignore_index=True)
    suffix = "_quick" if args.quick else ""
    arms.to_csv(tables / f"external_arms{suffix}.csv", index=False)
    contrasts.to_csv(tables / f"external_contrasts{suffix}.csv", index=False)
    figure(arms, figures / f"external{suffix}.png")

    # ---- Print the summary ----
    print(contrasts[contrasts["metric"] == "uno_c"].to_string(
        index=False, float_format=lambda v: f"{v:+.3f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
