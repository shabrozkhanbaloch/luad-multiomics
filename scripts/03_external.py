#!/usr/bin/env python
"""Step 3. External validation: models trained on TCGA, tested on GEO.

    python scripts/03_external.py
    python scripts/03_external.py --quick

Each GEO patient has expression and clinical data only, so mutation and
methylation are missing for real, not by simulation. Writes
external_arms.csv, external_contrasts.csv and a figure.
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

ARMS = ["clinical", "clinical+rna", "early", "late", "mofa"]
KEEP = ["clinical", "clinical+rna", "early@geo", "late@geo", "mofa@geo"]


def one(cfg, tcga_endpoint, geo_endpoint, cohort_id, quick, n_boot):
    cfg = copy.deepcopy(cfg)
    cfg["scenarios"] = {"full": [], "geo": ["mutation", "methylation"]}
    seed = int(cfg["project"]["seed"])
    tcga = load_cohort(cfg, tcga_endpoint, verbose=False)
    expr, clin, y_geo = load_geo(cfg, cohort_id, geo_endpoint)
    both, tr, te = stack(tcga, expr, clin, y_geo, cfg["external"]["shared_clinical"])
    res = run_fold(both, tr, te, f"{cohort_id}_{geo_endpoint}", cfg, seed, quick, 1,
                   Path(cfg["paths"]["cache"]) / "mofa", ARMS)

    arms_all, con_all = [], []
    for model, risks in res["risk"].items():
        # Uno's weights need the censoring distribution of the cohort being
        # scored, so the reference rows are the GEO cohort itself.
        fold = {"train": te, "test": te, "risk": {k: v for k, v in risks.items() if k in KEEP}}
        b = cfg["bootstrap"]
        arms, con = paired_bootstrap([fold], both.y, [tuple(x) for x in cfg["external"]["contrasts"]],
                                     n_boot, seed, int(b["min_events_per_resample"]),
                                     float(b["level"]))
        for df in (arms, con):
            df.insert(0, "model", model)
            df.insert(0, "n_events", int(y_geo["event"].sum()))
            df.insert(0, "n", len(te))
            df.insert(0, "endpoint", f"{tcga_endpoint.upper()}->{geo_endpoint.upper()}")
            df.insert(0, "cohort", cohort_id)
        arms_all.append(arms)
        con_all.append(con)
    return pd.concat(arms_all), pd.concat(con_all)


def figure(arms: pd.DataFrame, path) -> None:
    d = arms[arms["metric"] == "uno_c"]
    panels = list(d.groupby(["cohort", "endpoint"], sort=False))
    fig, axes = plt.subplots(1, len(panels), figsize=(3.6 * len(panels), 3.6), sharey=True)
    for ax, ((cid, ep), g) in zip(np_atleast(axes), panels):
        for mi, (model, gm) in enumerate(g.groupby("model", sort=False)):
            gm = gm.set_index("arm").loc[KEEP]
            ys = [i + (mi - 0.5) * 0.3 for i in range(len(KEEP))]
            ax.errorbar(gm["c_index"], ys, xerr=[gm["c_index"] - gm["ci_low"],
                                                 gm["ci_high"] - gm["c_index"]],
                        fmt="o", ms=4, lw=1.2, label=model, color=["#2a78d6", "#c2541c"][mi])
        ax.axvline(0.5, color="#898781", lw=0.8, ls="--")
        ax.set_yticks(range(len(KEEP)))
        ax.set_yticklabels(KEEP, fontsize=8)
        ax.set_title(f"{cid}\n{ep}  (n={g['n'].iloc[0]}, {g['n_events'].iloc[0]} events)", fontsize=9)
        ax.set_xlabel("Uno's C")
    np_atleast(axes)[0].legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def np_atleast(axes):
    import numpy as np
    return np.atleast_1d(axes)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--jobs", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config()
    tables, figures = out_dirs(cfg)
    n_boot = 200 if args.quick else int(cfg["bootstrap"]["n_resamples"])

    jobs = []
    for tcga_ep, geo_ep in cfg["external"]["endpoints"].items():
        for cid in cfg["external"]["cohorts"]:
            clin = pd.read_parquet(Path(cfg["paths"]["processed"]) / f"geo_{cid}_clinical.parquet")
            if clin[f"{geo_ep}_event"].sum() >= 20:        # the cohort records this endpoint
                jobs.append((tcga_ep.lower(), geo_ep, cid))
            else:
                print(f"  {cid}: no usable {geo_ep.upper()} endpoint, skipped")

    out = Parallel(n_jobs=min(n_jobs(args.jobs), len(jobs)), verbose=5)(
        delayed(one)(cfg, t, g, cid, args.quick, n_boot) for t, g, cid in jobs)
    arms = pd.concat([a for a, _ in out], ignore_index=True)
    con = pd.concat([c for _, c in out], ignore_index=True)
    suffix = "_quick" if args.quick else ""
    arms.to_csv(tables / f"external_arms{suffix}.csv", index=False)
    con.to_csv(tables / f"external_contrasts{suffix}.csv", index=False)
    figure(arms, figures / f"external{suffix}.png")
    print(con[con["metric"] == "uno_c"].to_string(index=False, float_format=lambda v: f"{v:+.3f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
