#!/usr/bin/env python
"""Step 5. The four robustness checks, fixed in advance.

    python scripts/05_checks.py                # all four
    python scripts/05_checks.py --skip-sweep   # without the width sweep (the slow one)

    1. Positive controls   the data are joined correctly and carry known biology
    2. Leakage             transforms do not read test rows; a shuffled outcome scores 0.50
    3. Capacity            the answer does not depend on how wide the omics blocks are
    4. Metric              verdicts under Uno's C against Harrell's C

Writes results/tables/checks.csv and width_sweep.csv.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import replace

from _common import load_config, n_jobs, out_dirs

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from luad import models as M
from luad.biology import cox_score_z
from luad.cv import outer_folds, run_fold
from luad.data import load_cohort
from luad.transforms import build_blocks

ROWS: list[dict] = []


def add(check: str, what: str, result: str, ok: bool | None, detail: str = "") -> None:
    verdict = "PASS" if ok else ("FAIL" if ok is False else "INFO")
    ROWS.append({"check": check, "what": what, "result": result, "verdict": verdict, "detail": detail})
    print(f"[{verdict}] {check}: {what}: {result}  {detail}")


def positive_controls(c, cfg):
    bio = cfg["biology"]
    male = c.clinical[:, c.clinical_names.index("sex_male")]
    rs = {g: float(np.corrcoef(c.rna[:, np.flatnonzero(c.rna_genes == g)[0]], male)[0, 1])
          for g in bio["sex_genes"] if (c.rna_genes == g).any()}
    best = max(rs.values()) if rs else float("nan")
    add("1 positive control", "expression rows belong to the right patients",
        f"r = {best:.2f}", bool(best >= 0.8),
        "Y-linked genes against recorded sex: " + ", ".join(f"{g} {r:+.2f}" for g, r in rs.items()))

    idx = [np.flatnonzero(c.rna_genes == g)[0] for g in bio["proliferation_metagene"]
           if (c.rna_genes == g).any()]
    score = c.rna[:, idx].mean(axis=1)
    z = float(cox_score_z(score[:, None], c.y)[0])
    add("1 positive control", "expression carries known prognostic biology",
        f"z = {z:+.2f}", bool(z > 2),
        f"proliferation metagene, {len(idx)} genes, no fitting; "
        f"Uno's C {M.uno_c(c.y, c.y, score):.3f}")


def leakage(c, cfg, seed, tables, endpoint):
    _, tr, te = outer_folds(c, cfg, seed, quick=True)[0]
    rng = np.random.default_rng(seed)
    bad = replace(c, rna=c.rna.copy(), meth=c.meth.copy(), mut_gene=c.mut_gene.copy(),
                  clinical=c.clinical.copy(), rna_counts=c.rna_counts.copy())
    bad.rna[te] = rng.random(bad.rna[te].shape, dtype=np.float32)
    bad.meth[te] = rng.random(bad.meth[te].shape, dtype=np.float32)
    bad.mut_gene[te] = 1 - bad.mut_gene[te]
    bad.clinical[te] = rng.normal(size=bad.clinical[te].shape)
    bad.rna_counts[te] = rng.poisson(500, bad.rna_counts[te].shape)
    a, b = build_blocks(c, tr, cfg, seed, 500), build_blocks(bad, tr, cfg, seed, 500)
    same = all(a[k].names == b[k].names and np.array_equal(a[k].X[tr], b[k].X[tr]) for k in a)
    add("2 leakage", "corrupting every test row changes nothing that was fitted",
        "identical" if same else "DIFFERENT", same, "all four blocks, real data, one fold")

    path = tables / f"cv_{endpoint}_permuted_summary.csv"
    if not path.exists():
        add("2 leakage", "shuffled outcome scores 0.50", "not run", None,
            "run: python scripts/01_cv.py --permute")
        return
    s = pd.read_csv(path)
    worst = s.loc[(s["uno_c"] - 0.5).abs().idxmax()]
    add("2 leakage", "shuffled outcome scores 0.50",
        f"mean {s['uno_c'].mean():.3f}, range {s['uno_c'].min():.3f} to {s['uno_c'].max():.3f}",
        bool((s["uno_c"] - 0.5).abs().max() < 0.05),
        f"{len(s)} arms; furthest from 0.50 is {worst['arm']} [{worst['model']}]")


def width_sweep(c, cfg, seed, tables, jobs):
    """Fix the expression/methylation width instead of tuning it."""
    arms = ["clinical", "clinical+rna", "early"]
    folds = outer_folds(c, cfg, seed)
    rows = []
    for k in cfg["features"]["k_sweep"]:
        cfg_k = copy.deepcopy(cfg)
        cfg_k["features"]["k_grid"] = [int(k)]
        res = Parallel(n_jobs=min(jobs, len(folds)))(
            delayed(run_fold)(c, tr, te, fid, cfg_k, seed, False, 1, None, arms)
            for fid, tr, te in folds)
        for r in res:
            for model, risks in r["risk"].items():
                base = M.uno_c(c.y[r["train"]], c.y[r["test"]], risks["clinical"])
                for arm in arms[1:]:
                    cidx = M.uno_c(c.y[r["train"]], c.y[r["test"]], risks[arm])
                    rows.append({"k": k, "model": model, "arm": arm, "fold": r["fold"],
                                 "uno_c": cidx, "minus_clinical": cidx - base})
        print(f"  width {k} done")
    df = pd.DataFrame(rows)
    out = (df.groupby(["model", "arm", "k"], sort=False)
             .agg(uno_c=("uno_c", "mean"), minus_clinical=("minus_clinical", "mean"),
                  sd_across_folds=("minus_clinical", "std")).reset_index())
    out.to_csv(tables / "width_sweep.csv", index=False)
    best = out["minus_clinical"].max()
    add("3 capacity", "no fixed width puts omics above clinical",
        f"best gain over clinical {best:+.3f}", None,
        "mean over 25 folds at widths " + ", ".join(map(str, cfg["features"]["k_sweep"]))
        + "; see width_sweep.csv")


def chosen_widths(tables, endpoint):
    path = tables / f"cv_{endpoint}_info.csv"
    if not path.exists():
        return
    info = pd.read_csv(path).dropna(subset=["k"])
    info = info[info["arm"] != "mofa"]
    share = info["k"].value_counts(normalize=True).sort_index()
    add("3 capacity", "width the inner loop chose",
        ", ".join(f"{int(k)}: {v:.0%}" for k, v in share.items()), None,
        f"{len(info)} tuned fits with an expression or methylation block")


def metric(tables, endpoint):
    path = tables / f"contrasts_{endpoint}.csv"
    if not path.exists():
        add("4 metric", "verdicts under Uno's C and Harrell's C", "not run", None,
            "run: python scripts/02_bootstrap.py")
        return
    con = pd.read_csv(path)
    con = con[con["part"] != "control"]
    wide = con.pivot_table(index=["model", "a", "b"], columns="metric",
                           values=["verdict", "difference"], aggfunc="first")
    agree = wide["verdict"]["uno_c"] == wide["verdict"]["harrell_c"]
    flips = (np.sign(wide["difference"]["uno_c"]) != np.sign(wide["difference"]["harrell_c"])).sum()
    differ = [f"{a} minus {b} [{m}]" for (m, a, b), ok in agree.items() if not ok]
    add("4 metric", "verdicts agree under Uno's C and Harrell's C",
        f"{int(agree.sum())} of {len(agree)} agree; {int(flips)} change sign", None,
        ("differ: " + "; ".join(differ)) if differ else "")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-sweep", action="store_true")
    ap.add_argument("--jobs", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config()
    seed = int(cfg["project"]["seed"])
    endpoint = cfg["endpoint"]["primary"].lower()
    tables, _ = out_dirs(cfg)
    c = load_cohort(cfg, endpoint)

    positive_controls(c, cfg)
    leakage(c, cfg, seed, tables, endpoint)
    chosen_widths(tables, endpoint)
    if args.skip_sweep:
        if (tables / "width_sweep.csv").exists():
            out = pd.read_csv(tables / "width_sweep.csv")
            add("3 capacity", "no fixed width puts omics above clinical",
                f"best gain over clinical {out['minus_clinical'].max():+.3f}", None,
                "from the existing width_sweep.csv")
    else:
        width_sweep(c, cfg, seed, tables, n_jobs(args.jobs))
    metric(tables, endpoint)

    pd.DataFrame(ROWS).to_csv(tables / "checks.csv", index=False)
    failed = [r for r in ROWS if r["verdict"] == "FAIL"]
    print(f"\n{len(ROWS)} results, {len(failed)} failed -> {tables}/checks.csv")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
