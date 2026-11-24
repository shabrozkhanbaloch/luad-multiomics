"""External validation: train on TCGA, test on a GEO cohort.

What this file is for
    A model that works inside one study may not work in another. External
    validation tests this: the models are trained on all TCGA patients and
    applied, unchanged, to patients of a different study from the GEO database
    (a public archive of expression data). scripts/03_external.py does this for
    three GEO cohorts, using the three functions here in this order:

        load_geo            reads one GEO cohort from data/processed.
        stack               puts TCGA and GEO into one cohort.
        within_sample_rank  used by `stack` to put the two expression tables
                            on one scale.

How it reuses the ordinary pipeline
    The GEO cohorts have expression and four clinical covariates and nothing
    else. So this is not a separate pipeline: the TCGA and GEO patients are
    stacked into one cohort, TCGA is the training fold, GEO is the test fold, and
    the ordinary fold runner (cv.run_fold) is called with mutation and
    methylation declared missing. Every transform is fitted on the training rows
    only, which the test suite checks, so the placeholder values in the GEO rows
    are never read.

Why ranks
    TCGA expression is sequencing counts and the GEO cohorts are microarrays, so
    the raw numbers are not comparable. Inside one patient, though, the order of
    the genes from lowest to highest is comparable. Each patient's genes are
    therefore replaced by their rank within that patient.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from .data import DAYS_PER_MONTH, Cohort, censor_at, make_y


def within_sample_rank(X: np.ndarray) -> np.ndarray:
    """Rank each patient's genes from lowest to highest, scaled to (0, 1].

    X  expression, patients x genes.

    Returns an array of the same shape (float32). In each row the lowest gene
    gets 1 / n_genes and the highest gets 1; genes with equal values share the
    average of their ranks. Platform-free: any monotone transform of a sample
    (one that keeps the order of its values, such as a logarithm) gives the
    same ranks.
    """
    return (rankdata(X, method="average", axis=1) / X.shape[1]).astype(np.float32)


def load_geo(cfg: dict, cohort_id: str, endpoint: str):
    """Read one GEO cohort, keeping the samples that have the endpoint.

    cfg        the configuration; gives the data folder and the censoring horizon.
    cohort_id  the GEO series, e.g. "GSE72094".
    endpoint   the GEO name of the endpoint, "rfs" or "os"; the clinical table
               has the columns "<endpoint>_time" and "<endpoint>_event".

    Returns (expression, clinical, y) for the same samples in the same order:
        expression  DataFrame, genes (rows, by gene symbol) x samples (columns).
        clinical    DataFrame, one row per sample.
        y           structured array (event, time in days), one entry per sample.
    GEO times are in months; they are converted to days and censored at the
    horizon, as the TCGA times are.
    """
    processed = Path(cfg["paths"]["processed"])
    expr = pd.read_parquet(processed / f"geo_{cohort_id}_expr.parquet").set_index("gene_name")
    clin = pd.read_parquet(processed / f"geo_{cohort_id}_clinical.parquet").set_index("sample_id")
    # Put the clinical rows in the order of the expression columns.
    clin = clin.loc[expr.columns]
    times = clin[f"{endpoint}_time"].to_numpy(float) * DAYS_PER_MONTH
    events = clin[f"{endpoint}_event"].to_numpy(float)
    # A sample is usable only if its time and its event status are both
    # recorded and the follow-up is longer than zero.
    keep = np.isfinite(times) & np.isfinite(events) & (times > 0)
    # Follow-up is cut at the same horizon as in TCGA (60 months): an event
    # after it is not counted, and the patient is recorded as event-free at
    # the horizon.
    horizon = float(cfg["endpoint"]["admin_censor_months"]) * DAYS_PER_MONTH
    event, time = censor_at(events[keep] == 1, times[keep], horizon)
    return expr.loc[:, keep], clin.loc[keep], make_y(event, time)


def stack(tcga: Cohort, expr: pd.DataFrame, clin: pd.DataFrame, y_geo: np.ndarray,
          shared_clinical: list[str]) -> tuple[Cohort, np.ndarray, np.ndarray]:
    """Put TCGA and one GEO cohort into a single cohort, TCGA rows first.

    tcga             the TCGA cohort.
    expr             GEO expression, genes x samples (from `load_geo`).
    clin             GEO clinical table, one row per sample (from `load_geo`).
    y_geo            GEO outcome, one entry per sample (from `load_geo`).
    shared_clinical  names of the clinical covariates both cohorts have.

    Returns (cohort, train, test). The cohort holds the TCGA patients and then
    the GEO samples; it keeps only the genes and clinical covariates the two
    have in common. train is the row numbers of the TCGA patients and test
    those of the GEO samples, ready to be passed to cv.run_fold.
    """
    # One TCGA column per gene symbol: the one with the highest mean count.
    # The columns are walked from the highest mean count down, and setdefault
    # keeps the first column seen for each symbol.
    order = np.argsort(-tcga.rna_counts.mean(axis=0), kind="stable")
    first_column = {}
    for column in order:
        first_column.setdefault(tcga.rna_genes[column], column)
    # The genes both cohorts measured, in alphabetical order, and the TCGA
    # column that holds each of them.
    shared = np.array(sorted(set(first_column) & set(expr.index)))
    cols = np.array([first_column[gene] for gene in shared])

    n_tcga, n_geo = tcga.n, expr.shape[1]
    geo_ranks = within_sample_rank(expr.loc[shared].to_numpy(np.float64).T)
    # Ranks are recomputed over the shared genes for both cohorts.
    tcga_ranks = within_sample_rank(tcga.rna_counts[:, cols].astype(np.float64))

    # Clinical covariates: the shared ones, in the same order for both cohorts.
    clinical_columns = [tcga.clinical_names.index(name) for name in shared_clinical]
    clinical = np.vstack([tcga.clinical[:, clinical_columns],
                          clin[shared_clinical].to_numpy(np.float64)])

    def pad(block):            # GEO has no such block; these rows are never read
        """Add one row of zeros per GEO sample under a TCGA table, so that
        every table of the stacked cohort has the same number of rows."""
        return np.vstack([block, np.zeros((n_geo, block.shape[1]), block.dtype)])

    both = Cohort(
        patients=np.concatenate([tcga.patients, expr.columns.to_numpy(str)]),
        y=np.concatenate([tcga.y, y_geo]),
        clinical=clinical, clinical_names=list(shared_clinical),
        rna=np.vstack([tcga_ranks, geo_ranks]),
        rna_counts=pad(tcga.rna_counts[:, cols]), rna_genes=shared,
        mut_gene=pad(tcga.mut_gene), mut_gene_names=tcga.mut_gene_names,
        mut_feat=pad(tcga.mut_feat), mut_feat_names=tcga.mut_feat_names,
        meth=pad(tcga.meth), meth_probes=tcga.meth_probes)
    return both, np.arange(n_tcga), np.arange(n_tcga, n_tcga + n_geo)
