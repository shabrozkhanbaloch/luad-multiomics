"""Paired, patient-level bootstrap for differences in concordance.

One draw of the whole cohort per replicate, shared by every fold and every
arm. With repeated cross-validation a patient sits in several test folds;
drawing each fold on its own would resample one person several independent
times and make the intervals too narrow. A patient drawn twice is counted
twice wherever they appear; a patient not drawn is absent everywhere.
"""
from __future__ import annotations

import zlib

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .models import harrell_c, uno_c

METRICS = ("uno_c", "harrell_c")


def cohort_draws(n_patients: int, n_boot: int, seed: int) -> np.ndarray:
    """(n_boot, n_patients) patient indices. crc32, not hash(): Python salts
    string hashing per process, which would change the draws on every run."""
    rng = np.random.default_rng([seed, zlib.crc32(b"cohort")])
    return rng.integers(0, n_patients, (n_boot, n_patients))


def _score(metric, y_tr, y_te, risk):
    return uno_c(y_tr, y_te, risk) if metric == "uno_c" else harrell_c(y_te, risk)


def _fold(fold: dict, y: np.ndarray, arms: list[str], draws: np.ndarray, min_events: int):
    """Point scores and bootstrap scores for one fold.

    fold = {"train": idx, "test": idx, "risk": {arm: risk aligned to test}}.
    Returns point (n_metrics, n_arms) and boot (n_boot, n_metrics, n_arms).
    """
    tr, te = fold["train"], fold["test"]
    y_tr = y[tr]
    point = np.full((len(METRICS), len(arms)), np.nan)
    for mi, metric in enumerate(METRICS):
        for j, a in enumerate(arms):
            if a in fold["risk"]:
                point[mi, j] = _score(metric, y_tr, y[te], fold["risk"][a])

    pos = np.full(len(y), -1, dtype=np.int64)
    pos[te] = np.arange(len(te))
    boot = np.full((draws.shape[0], len(METRICS), len(arms)), np.nan)
    for b in range(draws.shape[0]):
        sel = pos[draws[b]]
        idx = sel[sel >= 0]                      # drawn patients this fold tested
        if len(idx) < 2:
            continue
        y_te = y[te[idx]]
        if y_te["event"].sum() < min_events:
            continue
        for mi, metric in enumerate(METRICS):
            for j, a in enumerate(arms):
                if a in fold["risk"]:
                    boot[b, mi, j] = _score(metric, y_tr, y_te, fold["risk"][a][idx])
    return point, boot


def verdict(lo: float, hi: float) -> str:
    if lo > 0:
        return "a better"
    if hi < 0:
        return "b better"
    return "not resolved"


def paired_bootstrap(folds: list[dict], y: np.ndarray, contrasts: list[tuple[str, str]],
                     n_boot: int, seed: int, min_events: int = 10, level: float = 0.95,
                     jobs: int = 1) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (arms table, contrasts table), one row per arm/contrast and metric.

    The estimate is the mean over folds; each replicate is the same mean on
    one shared draw of the cohort.
    """
    arms = sorted({a for f in folds for a in f["risk"]})
    draws = cohort_draws(len(y), n_boot, seed)
    out = Parallel(n_jobs=jobs)(delayed(_fold)(f, y, arms, draws, min_events) for f in folds)
    point = np.nanmean(np.stack([p for p, _ in out]), axis=0)          # metrics x arms
    with np.errstate(all="ignore"):
        boot = np.nanmean(np.stack([b for _, b in out]), axis=0)       # boot x metrics x arms
    q = [(1 - level) / 2 * 100, (1 + level) / 2 * 100]
    col = {a: j for j, a in enumerate(arms)}

    arm_rows, con_rows = [], []
    for mi, metric in enumerate(METRICS):
        for a in arms:
            lo, hi = np.nanpercentile(boot[:, mi, col[a]], q)
            arm_rows.append({"arm": a, "metric": metric, "c_index": point[mi, col[a]],
                             "ci_low": lo, "ci_high": hi})
        for a, b in contrasts:
            if a not in col or b not in col:
                continue
            d = boot[:, mi, col[a]] - boot[:, mi, col[b]]
            lo, hi = np.nanpercentile(d, q)
            con_rows.append({"a": a, "b": b, "metric": metric,
                             "difference": point[mi, col[a]] - point[mi, col[b]],
                             "ci_low": lo, "ci_high": hi, "verdict": verdict(lo, hi)})
    return pd.DataFrame(arm_rows), pd.DataFrame(con_rows)
