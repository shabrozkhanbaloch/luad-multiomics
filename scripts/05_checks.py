#!/usr/bin/env python
"""Step 5. The four robustness checks, fixed in advance.

    python scripts/05_checks.py                # all four
    python scripts/05_checks.py --skip-sweep   # without the width sweep (the slow one)

    1. Positive controls   the data are joined correctly and carry known biology
    2. Leakage             transforms do not read test rows; a shuffled outcome scores 0.50
    3. Capacity            the answer does not depend on how wide the omics blocks are
    4. Metric              verdicts under Uno's C against Harrell's C

What it is for
    The main results are only worth reading if the analysis itself is sound.
    Each check tries to break the analysis in one known way and reports
    PASS, FAIL or INFO (a number to read, with no pass mark). The checks
    were chosen before the results were seen.

What it reads
    config/config.yaml                                 the plan
    data/processed/, data/interim/                     the four blocks and the endpoints
    results/tables/cv_<endpoint>_permuted_summary.csv  from `01_cv.py --permute` (check 2)
    results/tables/cv_<endpoint>_info.csv              from step 1 (check 3)
    results/tables/contrasts_<endpoint>.csv            from step 2 (check 4)
    results/tables/width_sweep.csv                     only with --skip-sweep, if it exists

    A check whose input table does not exist yet is reported as "not run" (or
    left out, for the chosen widths); it does not stop the script.

What it writes
    results/tables/checks.csv        one row per result: check, what, result, verdict, detail
    results/tables/width_sweep.csv   concordance at each fixed block width (not with --skip-sweep)

    The command ends with an error if any check is a FAIL.

How long it takes
    About 46 minutes on a 12-worker machine, mostly the width sweep.
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

# Every result of every check is collected here by add(), in the order the
# checks run, and written to checks.csv at the end.
ROWS: list[dict] = []


def add(check: str, what: str, result: str, ok: bool | None, detail: str = "") -> None:
    """Record one result: keep it for checks.csv and print it.

    Arguments
        check    which of the four checks this belongs to, for example "2 leakage"
        what     the statement that was tested, in words
        result   what was found, as short text
        ok       True for PASS, False for FAIL, None for INFO (no pass mark)
        detail   optional extra text: how it was measured, or what to run

    Returns
        Nothing. A row is added to ROWS and one line is printed.
    """
    verdict = "PASS" if ok else ("FAIL" if ok is False else "INFO")
    ROWS.append({"check": check, "what": what, "result": result, "verdict": verdict, "detail": detail})
    print(f"[{verdict}] {check}: {what}: {result}  {detail}")


def positive_controls(cohort, cfg):
    """Check 1: the data are joined correctly and carry known biology.

    Arguments
        cohort   the analysis cohort, as returned by load_cohort
        cfg      the plan; the gene lists are under `biology`

    Returns
        Nothing. Two results are recorded with add().
    """
    biology_settings = cfg["biology"]

    # First control. Genes on the Y chromosome are expressed in men only. If
    # the expression rows were matched to the wrong patients, their expression
    # would no longer follow the sex recorded in the clinical data.
    male = cohort.clinical[:, cohort.clinical_names.index("sex_male")]
    correlations = {
        gene: float(np.corrcoef(cohort.rna[:, np.flatnonzero(cohort.rna_genes == gene)[0]],
                                male)[0, 1])
        for gene in biology_settings["sex_genes"] if (cohort.rna_genes == gene).any()}
    best = max(correlations.values()) if correlations else float("nan")
    add("1 positive control", "expression rows belong to the right patients",
        f"r = {best:.2f}", bool(best >= 0.8),
        "Y-linked genes against recorded sex: "
        + ", ".join(f"{gene} {r:+.2f}" for gene, r in correlations.items()))

    # Second control. Fast-dividing tumours are known to do worse. The mean
    # of a fixed list of proliferation genes should therefore be linked to the
    # outcome without any model being fitted. If not, the outcome or the
    # expression data are suspect.
    gene_columns = [np.flatnonzero(cohort.rna_genes == gene)[0]
                    for gene in biology_settings["proliferation_metagene"]
                    if (cohort.rna_genes == gene).any()]
    score = cohort.rna[:, gene_columns].mean(axis=1)
    z = float(cox_score_z(score[:, None], cohort.y)[0])
    add("1 positive control", "expression carries known prognostic biology",
        f"z = {z:+.2f}", bool(z > 2),
        f"proliferation metagene, {len(gene_columns)} genes, no fitting; "
        f"Uno's C {M.uno_c(cohort.y, cohort.y, score):.3f}")


def leakage(cohort, cfg, seed, tables, endpoint):
    """Check 2: nothing fitted reads the test patients, and a shuffled outcome scores 0.50.

    Arguments
        cohort     the analysis cohort
        cfg        the plan
        seed       the random seed of the project
        tables     the folder of result tables
        endpoint   the endpoint, used to find the permuted summary of step 1

    Returns
        Nothing. Two results are recorded with add().
    """
    # First part. Take one fold and overwrite every test row of every block
    # with noise. Feature selection, imputation and scaling are fitted on the
    # training rows only, so the chosen features and the training values must
    # come out exactly the same as with the real data. Any difference means
    # that something looked at the test patients.
    _, train_rows, test_rows = outer_folds(cohort, cfg, seed, quick=True)[0]
    rng = np.random.default_rng(seed)
    # The arrays are copied first, so the real cohort is left untouched.
    corrupted = replace(cohort, rna=cohort.rna.copy(), meth=cohort.meth.copy(),
                        mut_gene=cohort.mut_gene.copy(), clinical=cohort.clinical.copy(),
                        rna_counts=cohort.rna_counts.copy())
    corrupted.rna[test_rows] = rng.random(corrupted.rna[test_rows].shape, dtype=np.float32)
    corrupted.meth[test_rows] = rng.random(corrupted.meth[test_rows].shape, dtype=np.float32)
    corrupted.mut_gene[test_rows] = 1 - corrupted.mut_gene[test_rows]
    corrupted.clinical[test_rows] = rng.normal(size=corrupted.clinical[test_rows].shape)
    corrupted.rna_counts[test_rows] = rng.poisson(500, corrupted.rna_counts[test_rows].shape)
    real_blocks = build_blocks(cohort, train_rows, cfg, seed, 500)
    corrupted_blocks = build_blocks(corrupted, train_rows, cfg, seed, 500)
    same = all(real_blocks[name].names == corrupted_blocks[name].names
               and np.array_equal(real_blocks[name].X[train_rows], corrupted_blocks[name].X[train_rows])
               for name in real_blocks)
    add("2 leakage", "corrupting every test row changes nothing that was fitted",
        "identical" if same else "DIFFERENT", same, "all four blocks, real data, one fold")

    # Second part. `01_cv.py --permute` reran the cross-validation with the
    # outcomes shuffled between patients. There is then nothing to learn, so
    # an honest pipeline scores about 0.50. A clearly higher score would mean
    # information is leaking from the test patients into the models.
    path = tables / f"cv_{endpoint}_permuted_summary.csv"
    if not path.exists():
        add("2 leakage", "shuffled outcome scores 0.50", "not run", None,
            "run: python scripts/01_cv.py --permute")
        return
    summary = pd.read_csv(path)
    worst = summary.loc[(summary["uno_c"] - 0.5).abs().idxmax()]
    add("2 leakage", "shuffled outcome scores 0.50",
        f"mean {summary['uno_c'].mean():.3f}, "
        f"range {summary['uno_c'].min():.3f} to {summary['uno_c'].max():.3f}",
        bool((summary["uno_c"] - 0.5).abs().max() < 0.05),
        f"{len(summary)} arms; furthest from 0.50 is {worst['arm']} [{worst['model']}]")


def width_sweep(cohort, cfg, seed, tables, jobs):
    """Fix the expression/methylation width instead of tuning it.

    Check 3, the slow part. In the main analysis the number of expression
    genes and methylation probes kept (the width) is tuned inside each fold.
    Here the whole cross-validation is rerun once per fixed width, to see
    whether any width would have put omics above the clinical model.

    Arguments
        cohort   the analysis cohort
        cfg      the plan; the widths to try are `features: k_sweep`
        seed     the random seed of the project
        tables   the folder of result tables
        jobs     number of worker processes

    Returns
        Nothing. width_sweep.csv is written and one result is recorded with add().
    """
    arms = ["clinical", "clinical+rna", "early"]
    # The same outer folds as step 1, so the numbers are comparable with it.
    folds = outer_folds(cohort, cfg, seed)
    rows = []
    for width in cfg["features"]["k_sweep"]:
        # A grid with a single value leaves the inner loop nothing to choose:
        # this is how the width is fixed. The plan is copied so the change
        # does not carry over to the next width.
        cfg_fixed_width = copy.deepcopy(cfg)
        cfg_fixed_width["features"]["k_grid"] = [int(width)]
        fold_results = Parallel(n_jobs=min(jobs, len(folds)))(
            delayed(run_fold)(cohort, train_rows, test_rows, fold_id, cfg_fixed_width, seed, False, 1,
                              None, arms)
            for fold_id, train_rows, test_rows in folds)
        for fold_result in fold_results:
            for model, risks in fold_result["risk"].items():
                # The clinical model is the baseline of the fold. Each omics
                # arm is stored with its gain over that baseline.
                base = M.uno_c(cohort.y[fold_result["train"]], cohort.y[fold_result["test"]],
                               risks["clinical"])
                for arm in arms[1:]:
                    concordance = M.uno_c(cohort.y[fold_result["train"]],
                                          cohort.y[fold_result["test"]], risks[arm])
                    rows.append({"k": width, "model": model, "arm": arm, "fold": fold_result["fold"],
                                 "uno_c": concordance, "minus_clinical": concordance - base})
        print(f"  width {width} done")
    per_fold = pd.DataFrame(rows)
    # Average over the folds: one row per model, arm and width.
    sweep = (per_fold.groupby(["model", "arm", "k"], sort=False)
                     .agg(uno_c=("uno_c", "mean"), minus_clinical=("minus_clinical", "mean"),
                          sd_across_folds=("minus_clinical", "std")).reset_index())
    sweep.to_csv(tables / "width_sweep.csv", index=False)
    best = sweep["minus_clinical"].max()
    add("3 capacity", "no fixed width puts omics above clinical",
        f"best gain over clinical {best:+.3f}", None,
        f"mean over {len(folds)} folds at widths " + ", ".join(map(str, cfg["features"]["k_sweep"]))
        + "; see width_sweep.csv")


def chosen_widths(tables, endpoint):
    """Check 3, the fast part: report which width the inner loop chose in step 1.

    Arguments
        tables     the folder of result tables
        endpoint   the endpoint, used to find cv_<endpoint>_info.csv

    Returns
        Nothing. One result is recorded with add(), or none if step 1 has
        not been run.
    """
    path = tables / f"cv_{endpoint}_info.csv"
    if not path.exists():
        return
    # Only fits that had a width to choose are counted: rows with no width
    # have no expression or methylation block. For MOFA the same column holds
    # the number of factors, not a width, so MOFA is left out.
    info = pd.read_csv(path).dropna(subset=["k"])
    info = info[info["arm"] != "mofa"]
    share = info["k"].value_counts(normalize=True).sort_index()
    add("3 capacity", "width the inner loop chose",
        ", ".join(f"{int(width)}: {fraction:.0%}" for width, fraction in share.items()), None,
        f"{len(info)} tuned fits with an expression or methylation block")


def metric(tables, endpoint):
    """Check 4: do the verdicts of step 2 depend on the choice of concordance metric?

    Arguments
        tables     the folder of result tables
        endpoint   the endpoint, used to find contrasts_<endpoint>.csv

    Returns
        Nothing. One result is recorded with add().
    """
    path = tables / f"contrasts_{endpoint}.csv"
    if not path.exists():
        add("4 metric", "verdicts under Uno's C and Harrell's C", "not run", None,
            "run: python scripts/02_bootstrap.py")
        return
    contrasts = pd.read_csv(path)
    # The control is zero under any metric, so it is left out of the count.
    contrasts = contrasts[contrasts["part"] != "control"]
    # Put the two metrics side by side: one row per model and contrast, with
    # the verdict and the difference under each metric.
    wide = contrasts.pivot_table(index=["model", "a", "b"], columns="metric",
                                 values=["verdict", "difference"], aggfunc="first")
    agree = wide["verdict"]["uno_c"] == wide["verdict"]["harrell_c"]
    flips = (np.sign(wide["difference"]["uno_c"]) != np.sign(wide["difference"]["harrell_c"])).sum()
    differ = [f"{arm_a} minus {arm_b} [{model}]"
              for (model, arm_a, arm_b), same_verdict in agree.items() if not same_verdict]
    add("4 metric", "verdicts agree under Uno's C and Harrell's C",
        f"{int(agree.sum())} of {len(agree)} agree; {int(flips)} change sign", None,
        ("differ: " + "; ".join(differ)) if differ else "")


def main() -> int:
    """Run the four checks in order and write checks.csv.

    Returns
        1 if any check failed, otherwise 0.
    """
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--skip-sweep", action="store_true",
                        help="do not rerun the width sweep; reuse width_sweep.csv if it exists")
    parser.add_argument("--jobs", type=int, default=None,
                        help="number of worker processes (default: most of the CPU threads)")
    args = parser.parse_args()

    # ---- Read the plan ----
    cfg = load_config()
    seed = int(cfg["project"]["seed"])
    endpoint = cfg["endpoint"]["primary"].lower()
    tables, _ = out_dirs(cfg)

    # ---- Load the data ----
    cohort = load_cohort(cfg, endpoint)

    # ---- Compute: the four checks, in the order of the table ----
    positive_controls(cohort, cfg)
    leakage(cohort, cfg, seed, tables, endpoint)
    chosen_widths(tables, endpoint)
    if args.skip_sweep:
        # The sweep is the slow part. When it is skipped, the result of an
        # earlier sweep is reported instead, if there is one.
        if (tables / "width_sweep.csv").exists():
            sweep = pd.read_csv(tables / "width_sweep.csv")
            add("3 capacity", "no fixed width puts omics above clinical",
                f"best gain over clinical {sweep['minus_clinical'].max():+.3f}", None,
                "from the existing width_sweep.csv")
    else:
        width_sweep(cohort, cfg, seed, tables, n_jobs(args.jobs))
    metric(tables, endpoint)

    # ---- Write the table ----
    pd.DataFrame(ROWS).to_csv(tables / "checks.csv", index=False)

    # ---- Print the summary ----
    failed = [row for row in ROWS if row["verdict"] == "FAIL"]
    print(f"\n{len(ROWS)} results, {len(failed)} failed -> {tables}/checks.csv")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
