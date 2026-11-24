"""External validation: train on TCGA, test on a GEO cohort.

The GEO cohorts have expression and four clinical covariates and nothing
else. So this is not a separate pipeline: the TCGA and GEO patients are
stacked into one cohort, TCGA is the training fold, GEO is the test fold, and
the ordinary fold runner is called with mutation and methylation declared
missing. Every transform is fitted on the training rows only, which the test
suite checks, so the placeholder values in the GEO rows are never read.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from .data import DAYS_PER_MONTH, Cohort, censor_at, make_y


def within_sample_rank(X: np.ndarray) -> np.ndarray:
    """Rank each patient's genes, scaled to (0, 1]. Platform-free: any
    monotone transform of a sample gives the same ranks."""
    return (rankdata(X, method="average", axis=1) / X.shape[1]).astype(np.float32)


def load_geo(cfg: dict, cohort_id: str, endpoint: str):
    """(expression genes x samples, clinical, y) for samples with the endpoint.
    GEO times are in months; converted to days and censored at the horizon."""
    p = Path(cfg["paths"]["processed"])
    expr = pd.read_parquet(p / f"geo_{cohort_id}_expr.parquet").set_index("gene_name")
    clin = pd.read_parquet(p / f"geo_{cohort_id}_clinical.parquet").set_index("sample_id")
    clin = clin.loc[expr.columns]
    t = clin[f"{endpoint}_time"].to_numpy(float) * DAYS_PER_MONTH
    e = clin[f"{endpoint}_event"].to_numpy(float)
    keep = np.isfinite(t) & np.isfinite(e) & (t > 0)
    horizon = float(cfg["endpoint"]["admin_censor_months"]) * DAYS_PER_MONTH
    event, time = censor_at(e[keep] == 1, t[keep], horizon)
    return expr.loc[:, keep], clin.loc[keep], make_y(event, time)


def stack(tcga: Cohort, expr: pd.DataFrame, clin: pd.DataFrame, y_geo: np.ndarray,
          shared_clinical: list[str]) -> tuple[Cohort, np.ndarray, np.ndarray]:
    """One cohort holding TCGA then GEO rows; returns it with (train, test)."""
    # One TCGA column per gene symbol: the one with the highest mean count.
    order = np.argsort(-tcga.rna_counts.mean(axis=0), kind="stable")
    first = {}
    for j in order:
        first.setdefault(tcga.rna_genes[j], j)
    shared = np.array(sorted(set(first) & set(expr.index)))
    cols = np.array([first[g] for g in shared])

    n_t, n_g = tcga.n, expr.shape[1]
    geo_ranks = within_sample_rank(expr.loc[shared].to_numpy(np.float64).T)
    # Ranks are recomputed over the shared genes for both cohorts.
    tcga_ranks = within_sample_rank(tcga.rna_counts[:, cols].astype(np.float64))

    ci = [tcga.clinical_names.index(n) for n in shared_clinical]
    clinical = np.vstack([tcga.clinical[:, ci], clin[shared_clinical].to_numpy(np.float64)])

    def pad(a):            # GEO has no such block; these rows are never read
        return np.vstack([a, np.zeros((n_g, a.shape[1]), a.dtype)])

    both = Cohort(
        patients=np.concatenate([tcga.patients, expr.columns.to_numpy(str)]),
        y=np.concatenate([tcga.y, y_geo]),
        clinical=clinical, clinical_names=list(shared_clinical),
        rna=np.vstack([tcga_ranks, geo_ranks]),
        rna_counts=pad(tcga.rna_counts[:, cols]), rna_genes=shared,
        mut_gene=pad(tcga.mut_gene), mut_gene_names=tcga.mut_gene_names,
        mut_feat=pad(tcga.mut_feat), mut_feat_names=tcga.mut_feat_names,
        meth=pad(tcga.meth), meth_probes=tcga.meth_probes)
    return both, np.arange(n_t), np.arange(n_t, n_t + n_g)
