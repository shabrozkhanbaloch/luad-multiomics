"""Fold-internal transforms.

What this file is for
    Turning the raw arrays of a `Cohort` into the feature matrices a model is
    fitted on. That means choosing which genes, probes and mutated genes to
    keep, and filling in missing clinical values.

The one rule of this module
    Every data-dependent choice (which genes, which probes, which mutated
    genes, the imputer) is fitted on the `train` rows and nothing else.

    Why it matters: in cross-validation the patients are split into training
    rows, which the model learns from, and test rows, which are used only to
    score it. If a choice such as "the most variable genes" were made on all
    patients, the test patients would have influenced the model, and the
    score would look better than it really is. That mistake is called
    leakage. `tests/test_leakage.py` checks that it does not happen here.

How the functions are used
    Each function returns the matrix for ALL rows, so the caller slices train
    and test itself and row order never changes. In every function:

    c       the `Cohort` (see data.py)
    train   the row numbers of the training patients, a 1-D integer array

Main functions
    build_blocks     the four blocks for one fold; the usual entry point
    concat           join several blocks side by side into one matrix
    impute_missing   blank out a block, as early fusion must when a patient lacks it

    The rest (`rna_block`, `methylation_block`, `mutation_block`,
    `clinical_block`, `top_variance`, `expressed_genes`) are the steps that
    `build_blocks` puts together.

"Width"
    The number of expression genes and of methylation probes kept is called
    the width, k. It is not fixed: cv.py tries several widths and keeps the
    best by inner cross-validation.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
# IterativeImputer is still marked experimental in scikit-learn; this import switches it on.
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer

from .data import Cohort


@dataclass
class Block:
    """One data block, ready for a model: a matrix and the name of each column.

    Column names carry a prefix that says which block they come from:
    `clin__`, `rna__`, `mut__` or `meth__`.
    """
    X: np.ndarray          # all patients x features, float32
    names: list[str]


def top_variance(X: np.ndarray, train: np.ndarray, k: int | None,
                 candidates: np.ndarray | None = None) -> np.ndarray:
    """Indices of the k most variable columns on the training rows.

    X
        patients x features.
    train
        Row numbers of the training patients.
    k
        How many columns to keep. None means keep all.
    candidates
        Column numbers to choose from. None means every column of X.

    Returns the chosen column numbers in increasing order. If there are k
    candidates or fewer, all of them are returned.
    """
    cols = np.arange(X.shape[1]) if candidates is None else candidates
    if k is None or len(cols) <= k:
        return cols

    # Variance is measured on training rows only (the rule of this module).
    var = X[train][:, cols].var(axis=0)
    # Largest variance first. A stable sort means that two columns with equal
    # variance always come out in the same order, so the run is reproducible.
    return np.sort(cols[np.argsort(-var, kind="stable")[:k]])


def expressed_genes(c: Cohort, train: np.ndarray, cfg: dict) -> np.ndarray:
    """Genes with CPM above a floor in enough training samples.

    CPM is counts per million: a gene's read count divided by the sample's
    total count, times one million. It makes samples sequenced to different
    depths comparable. A gene that is hardly detected in most samples carries
    noise and is dropped before the variance filter.

    cfg supplies the floor (`min_cpm`) and the share of training samples that
    must reach it (`min_expressed_prop`).

    Returns the column numbers of the genes that pass, in increasing order.
    """
    settings = cfg["preprocessing"]["expression"]
    counts = c.rna_counts[train].astype(np.float64)
    # Divide by each sample's total. The maximum with 1 avoids dividing by
    # zero for a sample with no counts at all.
    cpm = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1) * 1e6
    # For each gene: the share of training samples at or above the floor.
    return np.flatnonzero((cpm >= settings["min_cpm"]).mean(axis=0) >= settings["min_expressed_prop"])


def rna_block(c: Cohort, train: np.ndarray, cfg: dict, k: int | None) -> Block:
    """The expression block: the k most variable genes among the expressed ones.

    Both filters are fitted on the training rows. k is the width (None keeps
    every expressed gene). Returns a `Block` with patients x k columns.
    """
    gene_idx = top_variance(c.rna, train, k, expressed_genes(c, train, cfg))
    return Block(c.rna[:, gene_idx], [f"rna__{gene}" for gene in c.rna_genes[gene_idx]])


def methylation_block(c: Cohort, train: np.ndarray, k: int | None) -> Block:
    """The methylation block: the k most variable probes on the training rows.

    k is the width (None keeps every probe). Returns a `Block` with
    patients x k columns.
    """
    probe_idx = top_variance(c.meth, train, k)
    return Block(c.meth[:, probe_idx], [f"meth__{probe}" for probe in c.meth_probes[probe_idx]])


def mutation_block(c: Cohort, train: np.ndarray, cfg: dict) -> Block:
    """Recurrently mutated genes (chosen on train) + TMB + pathway indicators.

    Most genes are mutated in very few patients and cannot be learned from.
    A gene is kept if the share of training patients who carry a mutation in
    it reaches `gene_freq_threshold`. If more than `gene_max_features` genes
    pass, only the most often mutated are kept. Both settings come from cfg.

    Tumour mutation burden (TMB) and the pathway indicators are always kept.

    Returns a `Block`: patients x (kept genes + TMB + pathway indicators).
    """
    settings = cfg["preprocessing"]["mutation"]

    # How often each gene is mutated among the training patients.
    freq = c.mut_gene[train].mean(axis=0)
    keep = np.flatnonzero(freq >= settings["gene_freq_threshold"])

    # An optional upper limit on the number of genes: keep the most frequent.
    cap = settings.get("gene_max_features")
    if cap is not None and len(keep) > cap:
        keep = np.sort(keep[np.argsort(-freq[keep], kind="stable")[:cap]])

    # Gene columns first, then the features that are not filtered.
    X = np.hstack([c.mut_gene[:, keep], c.mut_feat]).astype(np.float32)
    names = ([f"mut__{gene}" for gene in c.mut_gene_names[keep]]
             + [f"mut__{feature}" for feature in c.mut_feat_names])
    return Block(X, names)


def clinical_block(c: Cohort, train: np.ndarray, seed: int) -> Block:
    """Iterative imputation fitted on train. Missingness indicators are
    per-patient facts and pass through unchanged.

    Some clinical values are not recorded for some patients. Iterative
    imputation predicts each covariate from the others and fills the gaps
    with the prediction. The relationships between covariates are learned
    from the training rows and then applied to every row.

    seed fixes the imputer's random choices so the result is reproducible.

    Returns a `Block`: patients x covariates, with no missing values.
    """
    imputer = IterativeImputer(random_state=seed, max_iter=20, sample_posterior=False)
    # Learn from the training rows only, then fill the gaps in all rows.
    imputer.fit(c.clinical[train])
    X = imputer.transform(c.clinical).astype(np.float32)
    return Block(X, [f"clin__{name}" for name in c.clinical_names])


def build_blocks(c: Cohort, train: np.ndarray, cfg: dict, seed: int,
                 k: int | None) -> dict[str, Block]:
    """All four blocks for one fold, at expression/methylation width k.

    c
        The cohort.
    train
        Row numbers of the training patients of this fold.
    cfg
        The configuration (filter settings).
    seed
        Seed for the clinical imputer.
    k
        The width: how many genes and how many probes to keep.

    Returns a dictionary from block name ("clinical", "rna", "mutation",
    "methylation") to `Block`. Every block has one row per patient of the
    cohort, training and test alike.
    """
    return {
        "clinical": clinical_block(c, train, seed),
        "rna": rna_block(c, train, cfg, k),
        "mutation": mutation_block(c, train, cfg),
        "methylation": methylation_block(c, train, k),
    }


def concat(blocks: dict[str, Block], which: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Design matrix for the named blocks, and each column's block label.

    A design matrix is the table a model is fitted on: one row per patient,
    one column per feature.

    blocks
        The output of `build_blocks`.
    which
        The block names to join, in the order their columns should appear.

    Returns (X, owner). X is patients x all the columns of the named blocks.
    owner has one entry per column of X: the name of the block that column
    came from. `impute_missing` uses it to find a block's columns.
    """
    X = np.hstack([blocks[name].X for name in which]).astype(np.float32)
    owner = np.concatenate([np.full(blocks[name].X.shape[1], name, dtype=object) for name in which])
    return X, owner


def impute_missing(X: np.ndarray, owner: np.ndarray, train: np.ndarray,
                   missing: list[str]) -> np.ndarray:
    """Marginal imputation: every column of a missing block takes its
    training mean. This is what early fusion has to do when a block is absent.

    An early-fusion model expects every column. When a patient has no data
    for a block, the least informative value to give is the average of the
    training patients, the same for everyone.

    X, owner
        The output of `concat`.
    train
        Row numbers of the training patients.
    missing
        The names of the blocks to treat as absent. An empty list means none.

    Returns a matrix of the same shape as X. The columns of the missing
    blocks are replaced in every row; the caller then uses the test rows. X
    itself is not modified (but with nothing missing, X itself is returned).
    """
    if not missing:
        return X
    out = X.copy()
    # The columns that belong to a missing block.
    cols = np.isin(owner, missing)
    # One mean per column, from training rows, written into every row.
    out[:, cols] = X[train][:, cols].mean(axis=0)
    return out
