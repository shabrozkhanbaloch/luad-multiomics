"""Fold-internal transforms.

The one rule of this module: every data-dependent choice (which genes, which
probes, which mutated genes, the imputer) is fitted on the `train` rows and
nothing else. Each function returns the matrix for ALL rows so the caller
slices train and test itself and row order never changes.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer

from .data import Cohort


@dataclass
class Block:
    X: np.ndarray          # all patients x features, float32
    names: list[str]


def top_variance(X: np.ndarray, train: np.ndarray, k: int | None,
                 candidates: np.ndarray | None = None) -> np.ndarray:
    """Indices of the k most variable columns on the training rows."""
    cols = np.arange(X.shape[1]) if candidates is None else candidates
    if k is None or len(cols) <= k:
        return cols
    var = X[train][:, cols].var(axis=0)
    return np.sort(cols[np.argsort(-var, kind="stable")[:k]])


def expressed_genes(c: Cohort, train: np.ndarray, cfg: dict) -> np.ndarray:
    """Genes with CPM above a floor in enough training samples."""
    e = cfg["preprocessing"]["expression"]
    counts = c.rna_counts[train].astype(np.float64)
    cpm = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1) * 1e6
    return np.flatnonzero((cpm >= e["min_cpm"]).mean(axis=0) >= e["min_expressed_prop"])


def rna_block(c: Cohort, train: np.ndarray, cfg: dict, k: int | None) -> Block:
    idx = top_variance(c.rna, train, k, expressed_genes(c, train, cfg))
    return Block(c.rna[:, idx], [f"rna__{g}" for g in c.rna_genes[idx]])


def methylation_block(c: Cohort, train: np.ndarray, k: int | None) -> Block:
    idx = top_variance(c.meth, train, k)
    return Block(c.meth[:, idx], [f"meth__{p}" for p in c.meth_probes[idx]])


def mutation_block(c: Cohort, train: np.ndarray, cfg: dict) -> Block:
    """Recurrently mutated genes (chosen on train) + TMB + pathway indicators."""
    m = cfg["preprocessing"]["mutation"]
    freq = c.mut_gene[train].mean(axis=0)
    keep = np.flatnonzero(freq >= m["gene_freq_threshold"])
    cap = m.get("gene_max_features")
    if cap is not None and len(keep) > cap:
        keep = np.sort(keep[np.argsort(-freq[keep], kind="stable")[:cap]])
    X = np.hstack([c.mut_gene[:, keep], c.mut_feat]).astype(np.float32)
    names = [f"mut__{g}" for g in c.mut_gene_names[keep]] + [f"mut__{n}" for n in c.mut_feat_names]
    return Block(X, names)


def clinical_block(c: Cohort, train: np.ndarray, seed: int) -> Block:
    """Iterative imputation fitted on train. Missingness indicators are
    per-patient facts and pass through unchanged."""
    imp = IterativeImputer(random_state=seed, max_iter=20, sample_posterior=False)
    imp.fit(c.clinical[train])
    X = imp.transform(c.clinical).astype(np.float32)
    return Block(X, [f"clin__{n}" for n in c.clinical_names])


def build_blocks(c: Cohort, train: np.ndarray, cfg: dict, seed: int,
                 k: int | None) -> dict[str, Block]:
    """All four blocks for one fold, at expression/methylation width k."""
    return {
        "clinical": clinical_block(c, train, seed),
        "rna": rna_block(c, train, cfg, k),
        "mutation": mutation_block(c, train, cfg),
        "methylation": methylation_block(c, train, k),
    }


def concat(blocks: dict[str, Block], which: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Design matrix for the named blocks, and each column's block label."""
    X = np.hstack([blocks[b].X for b in which]).astype(np.float32)
    owner = np.concatenate([np.full(blocks[b].X.shape[1], b, dtype=object) for b in which])
    return X, owner


def impute_missing(X: np.ndarray, owner: np.ndarray, train: np.ndarray,
                   missing: list[str]) -> np.ndarray:
    """Marginal imputation: every column of a missing block takes its
    training mean. This is what early fusion has to do when a block is absent."""
    if not missing:
        return X
    out = X.copy()
    cols = np.isin(owner, missing)
    out[:, cols] = X[train][:, cols].mean(axis=0)
    return out
