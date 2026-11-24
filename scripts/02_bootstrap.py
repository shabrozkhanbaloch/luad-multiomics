#!/usr/bin/env python
"""Step 2. Intervals for every arm and every pre-declared contrast.

    python scripts/02_bootstrap.py                # primary endpoint
    python scripts/02_bootstrap.py --endpoint os
    python scripts/02_bootstrap.py --quick        # 200 resamples

What it is for
    Step 1 gives one concordance per arm. This step says how certain each one
    is, and whether the difference between two arms (a contrast, "a minus b")
    is larger than chance. It uses a paired bootstrap: the patients are
    redrawn with replacement many times, and every arm is scored again on
    each redraw. The same redraw is used for every fold and every arm, so a
    difference between two arms is always measured on the same patients.
    Nothing is fitted here. The contrasts are the list in config.yaml and
    nothing else.

What it reads
    config/config.yaml                                the contrasts and the bootstrap settings
    results/tables/cv_<endpoint>_predictions.parquet  the risks step 1 wrote
    data/processed/, data/interim/                    the outcome of each patient

What it writes
    results/tables/arms_<endpoint>.csv         concordance of each arm, with its interval
    results/tables/contrasts_<endpoint>.csv    each contrast: difference, interval, verdict
    results/figures/contrasts_<endpoint>.png   a forest plot of the contrasts

    The command ends with an error if the pairing control is not exactly
    zero. The control is a contrast between two arms that give the same
    predictions (late fusion with every omics block missing, and clinical).

How long it takes
    About 13 minutes on a 12-worker machine.
"""
from __future__ import annotations

import argparse

from _common import load_config, load_predictions, n_jobs, out_dirs

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from luad.bootstrap import paired_bootstrap
from luad.data import load_cohort
from luad.models import MODEL_LABELS, MODELS

# A title for each group of contrasts in config.yaml. Nothing below uses it at present.
PART_LABELS = {"prediction": "Does a block add to clinical?",
               "integration": "Integration methods, nothing missing",
               "missing": "A block missing at test time",
               "control": "Control (must be exactly zero)"}


def forest(con: pd.DataFrame, path, title: str) -> None:
    """Draw the forest plot: one line per contrast, its interval and its difference.

    Arguments
        con     the contrasts table (columns a, b, model, metric, difference,
                ci_low, ci_high, verdict)
        path    the image file to write
        title   the title printed above the plot

    Returns
        Nothing. The plot is saved to `path`.
    """
    # Uno's C is the primary metric, so only its rows are drawn.
    uno_rows = con[con["metric"] == "uno_c"].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8.5, 0.3 * len(uno_rows) + 1.5))
    colours = {"a better": "#2a78d6", "b better": "#c2541c", "not resolved": "#898781"}
    for position, row in uno_rows.iterrows():
        # The first contrast is drawn at the top, so the height counts down.
        height = len(uno_rows) - 1 - position
        ax.plot([row.ci_low, row.ci_high], [height, height], color=colours[row.verdict], lw=2)
        ax.plot(row.difference, height, "o", color=colours[row.verdict], ms=5)
    # The vertical line at zero: an interval that crosses it is "not resolved".
    ax.axvline(0, color="#52514e", lw=0.8)
    ax.set_yticks(range(len(uno_rows)))
    ax.set_yticklabels([f"{row.a}  minus  {row.b}   [{row.model}]"
                        for row in uno_rows.itertuples()][::-1],
                       fontsize=7)
    ax.set_xlabel("Difference in Uno's C (95% interval, paired patient-level bootstrap)")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> int:
    """Bootstrap every arm and contrast for both models, then write the tables and the plot.

    Returns
        0 if the pairing control is exactly zero, otherwise 1.
    """
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--endpoint", default=None,
                        help="endpoint to analyse, pfi or os (default: the primary one in the config)")
    parser.add_argument("--quick", action="store_true",
                        help="200 resamples instead of the number in the config")
    parser.add_argument("--jobs", type=int, default=None,
                        help="number of worker processes (default: most of the CPU threads)")
    args = parser.parse_args()

    # ---- Read the plan ----
    cfg = load_config()
    endpoint = (args.endpoint or cfg["endpoint"]["primary"]).lower()
    tables, figures = out_dirs(cfg)
    bootstrap_settings = cfg["bootstrap"]
    n_boot = 200 if args.quick else int(bootstrap_settings["n_resamples"])

    # ---- Load the data ----
    # Only the outcomes are needed from the cohort: the risks come from step 1.
    cohort = load_cohort(cfg, endpoint, verbose=False)
    # The config groups the contrasts into parts. Here they become one flat
    # list of (part, arm a, arm b), in the order of the config.
    pairs = [(part, arm_a, arm_b) for part, pair_list in cfg["contrasts"].items()
             for arm_a, arm_b in pair_list]

    # ---- Compute: one bootstrap per model ----
    arms_all, contrasts_all = [], []
    for model in MODELS:
        folds = load_predictions(tables / f"cv_{endpoint}_predictions.parquet", model)
        arms, contrasts = paired_bootstrap(folds, cohort.y,
                                           [(arm_a, arm_b) for _, arm_a, arm_b in pairs], n_boot,
                                           int(cfg["project"]["seed"]),
                                           int(bootstrap_settings["min_events_per_resample"]),
                                           float(bootstrap_settings["level"]), n_jobs(args.jobs))
        # The bootstrap does not know which part a contrast belongs to, so the
        # part is looked up again and put in front, together with the model.
        part_of = {(arm_a, arm_b): part for part, arm_a, arm_b in pairs}
        contrasts.insert(0, "part", [part_of[(row.a, row.b)] for row in contrasts.itertuples()])
        contrasts.insert(1, "model", model)
        arms.insert(0, "model", model)
        arms_all.append(arms)
        contrasts_all.append(contrasts)
        print(f"{MODEL_LABELS[model]}: {len(folds)} folds, {n_boot} resamples")

    # ---- Write the tables ----
    arms = pd.concat(arms_all, ignore_index=True)
    contrasts = pd.concat(contrasts_all, ignore_index=True)
    # Sort the contrasts by part, in the order the parts have in the config
    # (not alphabetically), then by metric. The sort is stable, so within a
    # part and metric the rows keep the order they already had.
    order = {part: position for position, part in enumerate(cfg["contrasts"])}
    contrasts = contrasts.sort_values(
        ["part", "metric"], key=lambda s: s.map(order) if s.name == "part" else s,
        kind="stable").reset_index(drop=True)
    arms.to_csv(tables / f"arms_{endpoint}.csv", index=False)
    contrasts.to_csv(tables / f"contrasts_{endpoint}.csv", index=False)
    # The control is left out of the plot: it is zero by design and says
    # nothing about the data.
    forest(contrasts[contrasts["part"] != "control"], figures / f"contrasts_{endpoint}.png",
           f"Pre-declared contrasts, {endpoint.upper()}")

    # ---- Print the summary ----
    primary = contrasts[contrasts["metric"] == "uno_c"]
    print(primary.to_string(index=False, float_format=lambda v: f"{v:+.3f}"))
    # The pairing control compares two arms that give the same predictions
    # (late fusion with every omics block missing is the clinical model). If
    # the same patients are redrawn for both sides, the difference and both
    # ends of its interval are exactly zero. Anything else means the pairing
    # is broken.
    control = contrasts[contrasts["part"] == "control"]
    control_is_zero = bool(((control["difference"] == 0) & (control["ci_low"] == 0)
                            & (control["ci_high"] == 0)).all())
    print(f"\npairing control exactly zero: {'PASS' if control_is_zero else 'FAIL'}")
    return 0 if control_is_zero else 1


if __name__ == "__main__":
    raise SystemExit(main())
