#!/usr/bin/env python
"""Step 2. Intervals for every arm and every pre-declared contrast.

    python scripts/02_bootstrap.py                # primary endpoint
    python scripts/02_bootstrap.py --endpoint os
    python scripts/02_bootstrap.py --quick        # 200 resamples

Reads the predictions step 1 wrote; fits nothing. The contrasts are the list
in config.yaml and nothing else. Writes arms_<endpoint>.csv,
contrasts_<endpoint>.csv and a forest plot.
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

PART_LABELS = {"prediction": "Does a block add to clinical?",
               "integration": "Integration methods, nothing missing",
               "missing": "A block missing at test time",
               "control": "Control (must be exactly zero)"}


def forest(con: pd.DataFrame, path, title: str) -> None:
    d = con[con["metric"] == "uno_c"].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8.5, 0.3 * len(d) + 1.5))
    colours = {"a better": "#2a78d6", "b better": "#c2541c", "not resolved": "#898781"}
    for i, r in d.iterrows():
        yy = len(d) - 1 - i
        ax.plot([r.ci_low, r.ci_high], [yy, yy], color=colours[r.verdict], lw=2)
        ax.plot(r.difference, yy, "o", color=colours[r.verdict], ms=5)
    ax.axvline(0, color="#52514e", lw=0.8)
    ax.set_yticks(range(len(d)))
    ax.set_yticklabels([f"{r.a}  minus  {r.b}   [{r.model}]" for r in d.itertuples()][::-1],
                       fontsize=7)
    ax.set_xlabel("Difference in Uno's C (95% interval, paired patient-level bootstrap)")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--endpoint", default=None)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--jobs", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config()
    endpoint = (args.endpoint or cfg["endpoint"]["primary"]).lower()
    tables, figures = out_dirs(cfg)
    b = cfg["bootstrap"]
    n_boot = 200 if args.quick else int(b["n_resamples"])

    c = load_cohort(cfg, endpoint, verbose=False)
    pairs = [(part, a, bb) for part, lst in cfg["contrasts"].items() for a, bb in lst]

    arms_all, con_all = [], []
    for model in MODELS:
        folds = load_predictions(tables / f"cv_{endpoint}_predictions.parquet", model)
        arms, con = paired_bootstrap(folds, c.y, [(a, bb) for _, a, bb in pairs], n_boot,
                                     int(cfg["project"]["seed"]), int(b["min_events_per_resample"]),
                                     float(b["level"]), n_jobs(args.jobs))
        part_of = {(a, bb): part for part, a, bb in pairs}
        con.insert(0, "part", [part_of[(r.a, r.b)] for r in con.itertuples()])
        con.insert(1, "model", model)
        arms.insert(0, "model", model)
        arms_all.append(arms)
        con_all.append(con)
        print(f"{MODEL_LABELS[model]}: {len(folds)} folds, {n_boot} resamples")

    arms, con = pd.concat(arms_all, ignore_index=True), pd.concat(con_all, ignore_index=True)
    order = {p: i for i, p in enumerate(cfg["contrasts"])}
    con = con.sort_values(["part", "metric"], key=lambda s: s.map(order) if s.name == "part" else s,
                          kind="stable").reset_index(drop=True)
    arms.to_csv(tables / f"arms_{endpoint}.csv", index=False)
    con.to_csv(tables / f"contrasts_{endpoint}.csv", index=False)
    forest(con[con["part"] != "control"], figures / f"contrasts_{endpoint}.png",
           f"Pre-declared contrasts, {endpoint.upper()}")

    primary = con[con["metric"] == "uno_c"]
    print(primary.to_string(index=False, float_format=lambda v: f"{v:+.3f}"))
    ctrl = con[con["part"] == "control"]
    ok = bool(((ctrl["difference"] == 0) & (ctrl["ci_low"] == 0) & (ctrl["ci_high"] == 0)).all())
    print(f"\npairing control exactly zero: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
