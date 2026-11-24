"""Paired, patient-level bootstrap for differences in concordance.

What this file is for
    The cross-validation (cv.py) gives every test patient a risk from every
    arm. This file turns those risks into the numbers that are reported: the
    concordance of each arm, the difference between two arms (a "contrast"),
    and a 95% interval around each. scripts/02_bootstrap.py and
    scripts/03_external.py call `paired_bootstrap`; the other functions here
    are its parts.

Words used here
    concordance  (also "C" or "C-index") how well the risks put patients in the
                 right order. Take two patients: the pair is concordant if the
                 one with the higher risk has the event sooner. C is the share
                 of concordant pairs: 1.0 is a perfect order, 0.5 is chance.
                 Two versions are computed: Uno's C (the primary metric, which
                 corrects for patients lost to follow-up) and Harrell's C.
    bootstrap    a way to get an interval without a formula. Draw as many
                 patients as the cohort has, at random and WITH replacement, so
                 some are drawn twice and some not at all, and compute the
                 number again on that draw. Repeat many times (each repeat is a
                 "replicate"). The middle 95% of the results is the interval.
    paired       both arms of a contrast are scored on the same draw in every
                 replicate. What the two arms share (the luck of which patients
                 were drawn) then cancels in the difference, and the interval
                 is about the difference itself.
    resolved     an interval for a difference that does not contain zero.

Why patients are drawn once for the whole cohort
    There is one draw of the whole cohort per replicate, shared by every fold
    and every arm. With repeated cross-validation a patient sits in several
    test folds; drawing each fold on its own would resample one person several
    independent times and make the intervals too narrow. A patient drawn twice
    is counted twice wherever they appear; a patient not drawn is absent
    everywhere.
"""
from __future__ import annotations

import zlib

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .models import harrell_c, uno_c

# The two versions of concordance, in the order they are stored along the
# "metric" axis of the arrays below.
METRICS = ("uno_c", "harrell_c")


def cohort_draws(n_patients: int, n_boot: int, seed: int) -> np.ndarray:
    """The bootstrap draws: for each replicate, which patients were drawn.

    n_patients  size of the cohort.
    n_boot      number of replicates.
    seed        random seed; the same seed always gives the same draws.

    Returns an integer array of shape (n_boot, n_patients). Row b lists the row
    numbers of the patients drawn in replicate b; a number can appear more than
    once.

    The seed is mixed with crc32 of a fixed word, not with hash(): Python salts
    string hashing per process, which would change the draws on every run.
    """
    rng = np.random.default_rng([seed, zlib.crc32(b"cohort")])
    return rng.integers(0, n_patients, (n_boot, n_patients))


def _score(metric, y_tr, y_te, risk):
    """One concordance value.

    metric  "uno_c" or "harrell_c".
    y_tr    outcome of the training patients. Only Uno's C uses it, to estimate
            how patients are lost to follow-up.
    y_te    outcome of the patients being scored.
    risk    their risks, in the same order as y_te.
    """
    return uno_c(y_tr, y_te, risk) if metric == "uno_c" else harrell_c(y_te, risk)


def _fold(fold: dict, y: np.ndarray, arms: list[str], draws: np.ndarray, min_events: int):
    """Score one fold: once on its real test patients, once per replicate.

    fold        {"train": row numbers, "test": row numbers,
                 "risk": {arm: risks, one per test patient, in the order of "test"}}.
    y           outcome of every patient in the cohort (structured: event, time).
    arms        the arm names, which fix the order of the last axis of the output.
    draws       (n_boot, n_patients) row numbers, from `cohort_draws`.
    min_events  a replicate is skipped for this fold if fewer of the drawn test
                patients than this had an event; concordance on a handful of
                events is too unstable to use.

    Returns (point, boot). point has shape (n_metrics, n_arms): the scores on
    the real test patients. boot has shape (n_boot, n_metrics, n_arms): the
    scores in each replicate. A cell is NaN where the fold has no risks for
    that arm or the replicate was skipped.
    """
    train_rows, test_rows = fold["train"], fold["test"]
    y_train = y[train_rows]

    # The estimate itself: each arm scored on the fold's test patients as they are.
    point = np.full((len(METRICS), len(arms)), np.nan)
    for metric_index, metric in enumerate(METRICS):
        for arm_index, arm in enumerate(arms):
            if arm in fold["risk"]:
                point[metric_index, arm_index] = _score(metric, y_train, y[test_rows],
                                                        fold["risk"][arm])

    # The risks are stored in the order of the test rows. This lookup table
    # turns a patient's row number in the cohort into their position in that
    # order, or -1 for a patient who is not in this fold's test set.
    position_in_test = np.full(len(y), -1, dtype=np.int64)
    position_in_test[test_rows] = np.arange(len(test_rows))

    boot = np.full((draws.shape[0], len(METRICS), len(arms)), np.nan)
    for replicate in range(draws.shape[0]):
        # Of the patients drawn in this replicate, keep those this fold tested.
        # One drawn twice appears twice. The training rows are not resampled.
        drawn_positions = position_in_test[draws[replicate]]
        drawn = drawn_positions[drawn_positions >= 0]
        if len(drawn) < 2:
            continue
        y_drawn = y[test_rows[drawn]]
        if y_drawn["event"].sum() < min_events:
            continue
        # Every arm is scored on the same drawn patients: this is the pairing.
        for metric_index, metric in enumerate(METRICS):
            for arm_index, arm in enumerate(arms):
                if arm in fold["risk"]:
                    boot[replicate, metric_index, arm_index] = _score(
                        metric, y_train, y_drawn, fold["risk"][arm][drawn])
    return point, boot


def verdict(lo: float, hi: float) -> str:
    """Put an interval for the difference a minus b into words.

    lo, hi  the two ends of the interval.

    Returns "a better" if the whole interval is above zero, "b better" if it is
    all below zero, and "not resolved" if it contains zero.
    """
    if lo > 0:
        return "a better"
    if hi < 0:
        return "b better"
    return "not resolved"


def paired_bootstrap(folds: list[dict], y: np.ndarray, contrasts: list[tuple[str, str]],
                     n_boot: int, seed: int, min_events: int = 10, level: float = 0.95,
                     jobs: int = 1) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Concordance of every arm and the difference for every contrast, with
    bootstrap intervals.

    folds       one dict per fold, as described in `_fold`.
    y           outcome of every patient in the cohort (structured: event, time).
    contrasts   pairs of arm names (a, b); the difference reported is a minus b.
                A pair naming an arm that no fold has is left out.
    n_boot      number of bootstrap replicates.
    seed        random seed of the draws.
    min_events  fewest events a fold needs in a replicate to be scored.
    level       coverage of the intervals; 0.95 gives 95% intervals.
    jobs        number of folds scored at the same time.

    Returns (arms table, contrasts table), one row per arm or contrast and metric.
        arms table       columns arm, metric, c_index, ci_low, ci_high.
        contrasts table  columns a, b, metric, difference, ci_low, ci_high, verdict.

    The estimate is the mean over folds; each replicate is the same mean on
    one shared draw of the cohort.
    """
    # Every arm that appears in any fold, in alphabetical order.
    arms = sorted({arm for fold in folds for arm in fold["risk"]})
    # One set of draws, made before the folds are scored and given to all of
    # them: this is what makes the bootstrap patient-level.
    draws = cohort_draws(len(y), n_boot, seed)
    per_fold = Parallel(n_jobs=jobs)(delayed(_fold)(fold, y, arms, draws, min_events)
                                     for fold in folds)

    # Average over the folds, leaving out the cells that are NaN.
    point = np.nanmean(np.stack([fold_point for fold_point, _ in per_fold]), axis=0)   # metrics x arms
    # A replicate that every fold skipped has nothing to average and stays NaN;
    # the warning numpy would print for it is silenced.
    with np.errstate(all="ignore"):
        boot = np.nanmean(np.stack([fold_boot for _, fold_boot in per_fold]),
                          axis=0)                                           # boot x metrics x arms

    # The two percentiles that cut off the interval: 2.5 and 97.5 for 95%.
    percentiles = [(1 - level) / 2 * 100, (1 + level) / 2 * 100]
    column = {arm: arm_index for arm_index, arm in enumerate(arms)}

    arm_rows, contrast_rows = [], []
    for metric_index, metric in enumerate(METRICS):
        for arm in arms:
            lo, hi = np.nanpercentile(boot[:, metric_index, column[arm]], percentiles)
            arm_rows.append({"arm": arm, "metric": metric,
                             "c_index": point[metric_index, column[arm]],
                             "ci_low": lo, "ci_high": hi})
        for a, b in contrasts:
            if a not in column or b not in column:
                continue
            # The difference is taken replicate by replicate, a and b on the
            # same draw, and the interval is read from those differences.
            differences = boot[:, metric_index, column[a]] - boot[:, metric_index, column[b]]
            lo, hi = np.nanpercentile(differences, percentiles)
            contrast_rows.append({"a": a, "b": b, "metric": metric,
                                  "difference": (point[metric_index, column[a]]
                                                 - point[metric_index, column[b]]),
                                  "ci_low": lo, "ci_high": hi, "verdict": verdict(lo, hi)})
    return pd.DataFrame(arm_rows), pd.DataFrame(contrast_rows)
