"""Two survival model families, tuned by inner cross-validation.

What this file is for
    Given a feature matrix and the outcome, choose the settings of a model,
    fit it, and score its predictions. Everything else in the project (which
    blocks go into the matrix, which patients are missing what) is decided by
    the caller.

Risk
    A survival model gives each patient one number, a risk. Higher risk
    always means worse prognosis. The number has no unit; only the ordering
    of patients matters.

Concordance
    The score. Take two patients where one is known to have had the event
    before the other. The pair is "concordant" if the model gave the higher
    risk to the one who had the event first. Concordance (C) is the share of
    such pairs the model orders correctly: 0.5 is a coin toss, 1.0 is
    perfect. Harrell's C counts the pairs directly. Uno's C, the primary
    metric, also corrects for the pairs lost to censoring (patients last seen
    without the event), by weighting each pair.

The two model families
    coxnet   Penalised Cox regression. A linear model with a penalty that
             shrinks coefficients, many of them to exactly zero, so it can be
             fitted with more features than patients. `alpha` is the strength
             of the penalty and `l1_ratio` the kind (1 is lasso, lower values
             mix in ridge).
    rsf      Random survival forest. Many decision trees, each grown on a
             random resample; it can pick up effects that are not linear.

Inner cross-validation
    Settings such as alpha (called hyperparameters) cannot be chosen on the
    test rows, because that would be leakage. Instead the training rows of a
    fold are split again, into inner-training and inner-test parts. Each
    candidate setting is fitted on the inner-training parts and scored on the
    inner-test parts, and the setting with the best mean score is kept. The
    test rows of the outer fold are never touched.

Main functions
    inner_splits    make the inner split of the training rows
    tune            choose the best setting by inner cross-validation
    fit             refit that setting on all the training rows
    tune_and_fit    the two together
    uno_c, harrell_c   the two concordance scores

    A fitted model is returned as a `Fitted` with a `predict` function, so one
    fit can score many versions of the test rows (for example with a block
    removed).

Arguments that mean the same thing everywhere in this file
    X        patients x features, for ALL patients of the cohort
    y        the outcome of all patients, a structured array with the fields
             "event" and "time" (see data.make_y)
    train    row numbers of the training patients, a 1-D integer array
    splits   the inner split: a list of (inner-training rows, inner-test rows)
    cfg      the configuration; the grids of settings to try are under `models`
    seed     fixes the random choices
    threads  how many processor cores the forest may use
    quick    True for a fast trial run with a much smaller search
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Callable

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sksurv.ensemble import RandomSurvivalForest
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.metrics import concordance_index_censored, concordance_index_ipcw

MODELS = ("coxnet", "rsf")
MODEL_LABELS = {"coxnet": "Penalised Cox", "rsf": "Random survival forest"}


# ---------------------------------------------------------------- metrics
def uno_c(y_train, y_test, risk) -> float:
    """Uno's IPCW concordance, the primary metric. The censoring distribution
    comes from the training rows. NaN where the estimator is undefined.

    IPCW is inverse probability of censoring weighting: each pair of patients
    is weighted by how likely it was to stay uncensored, so that the score
    does not depend on how long patients happened to be followed.

    y_train
        Outcome of the training patients. Used only to estimate how censoring
        happens over time.
    y_test
        Outcome of the patients being scored.
    risk
        One risk per patient in y_test, higher meaning worse.

    Returns the concordance, or NaN if it cannot be computed.
    """
    # Only pairs whose earlier time is before tau are counted. Beyond the
    # last follow-up time seen in either set the weights are not defined.
    tau = min(y_train["time"].max(), y_test["time"].max())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return float(concordance_index_ipcw(y_train, y_test, risk, tau=tau)[0])
        except (ValueError, ZeroDivisionError):
            return float("nan")


def harrell_c(y_test, risk) -> float:
    """Harrell's concordance, reported beside Uno's C.

    y_test is the outcome of the patients being scored and risk is one number
    per patient, higher meaning worse. Returns the concordance, or NaN if it
    cannot be computed (for example when no patient had the event).
    """
    try:
        return float(concordance_index_censored(y_test["event"], y_test["time"], risk)[0])
    except (ValueError, ZeroDivisionError):
        return float("nan")


def _score(y_train, y_test, risk) -> float:
    """The score used to compare settings in inner cross-validation.

    Uno's C where it can be computed, otherwise Harrell's C, so that a small
    inner fold where Uno's C is undefined still gives a number.
    """
    score = uno_c(y_train, y_test, risk)
    return harrell_c(y_test, risk) if np.isnan(score) else score


def inner_splits(y, train: np.ndarray, n_folds: int, seed: int):
    """Split the training rows of one fold into inner folds.

    y
        Outcome of all patients.
    train
        Row numbers of the training patients.
    n_folds
        How many inner folds.
    seed
        Fixes the split.

    Returns a list of n_folds pairs (inner-training rows, inner-test rows).
    The rows are numbered in the whole cohort, like `train`, so they can be
    used directly on X and y. The split is stratified: each inner fold has
    about the same share of events.
    """
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    # The splitter returns positions within `train`; train[...] turns them
    # back into row numbers of the cohort.
    return [(train[inner_train_pos], train[inner_test_pos])
            for inner_train_pos, inner_test_pos in cv.split(train, y["event"][train].astype(int))]


# ---------------------------------------------------------------- fitted model
@dataclass
class Fitted:
    """A fitted model and what is needed to use and describe it.

    `predict` takes a matrix with the same columns the model was fitted on
    (any rows) and returns one risk per row.
    """
    model: str                              # "coxnet" or "rsf"
    params: dict                            # the chosen setting
    inner_score: float                      # mean inner-CV C of the chosen setting
    predict: Callable[[np.ndarray], np.ndarray]
    train_risk: np.ndarray                  # risk of the training rows
    selected: np.ndarray | None = None      # columns with non-zero coefficient (coxnet)


def _scaler(X, train):
    """A function that standardises columns using the training rows.

    Each column has its training mean subtracted and is divided by its
    training standard deviation. The penalty of the Cox model treats every
    coefficient alike, so the features must be on one scale first.

    Returns a function: give it a matrix with the same columns as X and it
    returns the standardised matrix (float64).
    """
    mu, sd = X[train].mean(axis=0), X[train].std(axis=0)
    # A column that is constant on the training rows has SD zero. Dividing
    # by 1 instead leaves it at zero rather than producing infinities.
    sd[sd < 1e-12] = 1.0
    return lambda A: ((A - mu) / sd).astype(np.float64)


# ---------------------------------------------------------------- penalised Cox
def _tune_coxnet(X, y, train, splits, cfg, quick):
    """One regularisation path per (l1_ratio, inner fold), scored at every
    alpha. The alpha grid comes from the outer-training rows so a given alpha
    means the same penalty in every inner fold.

    A regularisation path is the same model fitted at a whole series of
    penalty strengths (alphas) in one go, from strong to weak.

    Returns (best, best_score). best is a dictionary with the chosen
    `l1_ratio`, the chosen `alpha`, and `alphas`, the grid the alpha was
    chosen from (needed to refit the same path). best_score is its mean
    inner-CV concordance. best is None if no fit converged.
    """
    grid = cfg["models"]["coxnet"]
    # A quick run tries one l1_ratio and a short alpha grid.
    ratios = grid["l1_ratio"][:1] if quick else grid["l1_ratio"]
    n_alphas = 15 if quick else int(grid["n_alphas"])
    X_scaled = _scaler(X, train)(X)
    best, best_score = None, -np.inf
    for l1_ratio in ratios:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Fit once on all the training rows, only to let the library
            # choose the grid of alphas for this l1_ratio.
            try:
                path = CoxnetSurvivalAnalysis(l1_ratio=l1_ratio, n_alphas=n_alphas,
                                              alpha_min_ratio=0.01).fit(X_scaled[train], y[train])
            except (ValueError, ArithmeticError):
                continue
            alphas = np.asarray(path.alphas_, float)
            # The ends of the path are the null model and a barely penalised
            # fit that rarely converges when features outnumber patients.
            alphas = alphas[len(alphas) // 10: -1] if len(alphas) > 12 else alphas
            if not len(alphas):
                continue

            # For each alpha: the sum of its inner-fold scores, and how many
            # inner folds gave a usable score.
            total, count = np.zeros(len(alphas)), np.zeros(len(alphas))
            for inner_train, inner_test in splits:
                # The scaler is refitted on the inner-training rows.
                X_inner = _scaler(X, inner_train)(X)
                try:
                    inner_model = CoxnetSurvivalAnalysis(l1_ratio=l1_ratio, alphas=alphas).fit(
                        X_inner[inner_train], y[inner_train])
                except (ValueError, ArithmeticError):
                    continue
                for alpha_idx, alpha in enumerate(alphas):
                    risk = inner_model.predict(X_inner[inner_test], alpha=alpha)
                    # A fit that did not converge can give infinite or
                    # missing risks; such a fold is not counted for this alpha.
                    if np.all(np.isfinite(risk)):
                        total[alpha_idx] += _score(y[inner_train], y[inner_test], risk)
                        count[alpha_idx] += 1
        if not (count > 0).any():
            continue

        # Mean score per alpha. An alpha with no usable fold can never win.
        mean = np.where(count > 0, total / np.maximum(count, 1), -np.inf)
        alpha_idx = int(np.argmax(mean))
        # Keep this l1_ratio only if it beats the best one so far.
        if mean[alpha_idx] > best_score:
            best_score = float(mean[alpha_idx])
            best = {"l1_ratio": l1_ratio, "alpha": float(alphas[alpha_idx]), "alphas": alphas}
    return best, best_score


def _fit_coxnet(X, y, train, params, score) -> Fitted:
    """Refit the chosen penalised Cox setting on all the training rows.

    params is the dictionary returned by `_tune_coxnet` and score its mean
    inner-CV concordance, which is stored in the result unchanged.

    Returns a `Fitted`. Its `predict` standardises a matrix with the training
    means and SDs before scoring it. `selected` lists the columns whose
    coefficient is not zero: the features the model actually uses.
    """
    scale = _scaler(X, train)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # The whole path is refitted, as in tuning, and read at the chosen alpha.
        cox = CoxnetSurvivalAnalysis(l1_ratio=params["l1_ratio"],
                                     alphas=params["alphas"]).fit(scale(X[train]), y[train])
    alpha = params["alpha"]
    # The position on the path of the chosen alpha, to read its coefficients.
    alpha_idx = int(np.argmin(np.abs(np.asarray(cox.alphas_) - alpha)))
    predict = lambda A: cox.predict(scale(A), alpha=alpha)  # noqa: E731
    return Fitted("coxnet", params, score, predict, predict(X[train]),
                  np.flatnonzero(cox.coef_[:, alpha_idx] != 0))


# ---------------------------------------------------------------- random survival forest
def _rsf(params, n_trees, seed, threads):
    """An unfitted random survival forest with the given setting.

    params holds `max_features` (how many features each split may choose
    from) and `min_samples_leaf` (the smallest number of patients allowed in
    a final group). A node is split only if it holds at least twice that, so
    that both halves can be large enough.
    """
    return RandomSurvivalForest(
        n_estimators=n_trees, max_features=params["max_features"],
        min_samples_leaf=params["min_samples_leaf"],
        min_samples_split=2 * params["min_samples_leaf"],
        n_jobs=threads, random_state=seed)


def _tune_rsf(X, y, train, splits, cfg, seed, threads, quick):
    """Choose the forest's leaf size by inner cross-validation.

    Only `min_samples_leaf` is searched; `max_features` is taken from the
    configuration as it is. Tuning uses fewer trees than the final fit, to
    save time. `train` is not used: the inner folds in `splits` already say
    which rows to fit and score on.

    Returns (best, best_score): the chosen setting as a dictionary and its
    mean inner-CV concordance.
    """
    grid = cfg["models"]["rsf"]
    # A quick run tries one leaf size and a small forest.
    leaves = grid["min_samples_leaf"][:1] if quick else grid["min_samples_leaf"]
    n_trees = 50 if quick else int(grid["n_estimators_tune"])
    best, best_score = None, -np.inf
    for leaf in leaves:
        params = {"max_features": grid["max_features"], "min_samples_leaf": leaf}
        # Fit on each inner-training part, score on the matching inner-test part.
        scores = []
        for inner_train, inner_test in splits:
            risk = _rsf(params, n_trees, seed, threads).fit(X[inner_train], y[inner_train]).predict(
                X[inner_test])
            scores.append(_score(y[inner_train], y[inner_test], risk))
        mean_score = float(np.mean(scores))
        if mean_score > best_score:
            best, best_score = params, mean_score
    return best, best_score


def _fit_rsf(X, y, train, params, score, cfg, seed, threads, quick) -> Fitted:
    """Refit the chosen forest setting on all the training rows.

    The final forest has more trees than the ones used for tuning. A forest
    needs no scaling of its features and has no list of selected columns.
    Returns a `Fitted`.
    """
    n_trees = 100 if quick else int(cfg["models"]["rsf"]["n_estimators_final"])
    forest = _rsf(params, n_trees, seed, threads).fit(X[train], y[train])
    return Fitted("rsf", params, score, forest.predict, forest.predict(X[train]))


# ---------------------------------------------------------------- public
def tune(model: str, X: np.ndarray, y, train: np.ndarray, splits, cfg: dict,
         seed: int, threads: int = 1, quick: bool = False):
    """Best setting by inner CV: (params, mean inner C). params is None if
    nothing converged.

    model is "coxnet" or "rsf"; the other arguments are described at the top
    of this file. Nothing is fitted on the whole of `train` for keeps: pass
    the result to `fit` for that.
    """
    if model == "coxnet":
        return _tune_coxnet(X, y, train, splits, cfg, quick)
    if model == "rsf":
        return _tune_rsf(X, y, train, splits, cfg, seed, threads, quick)
    raise ValueError(f"unknown model {model!r}")


def fit(model: str, X: np.ndarray, y, train: np.ndarray, params: dict, score: float,
        cfg: dict, seed: int, threads: int = 1, quick: bool = False) -> Fitted:
    """Refit the chosen setting on all of `train`.

    params and score are the two values returned by `tune`. The other
    arguments are described at the top of this file. Returns a `Fitted`.
    """
    if model == "coxnet":
        return _fit_coxnet(X, y, train, params, score)
    if model == "rsf":
        return _fit_rsf(X, y, train, params, score, cfg, seed, threads, quick)
    raise ValueError(f"unknown model {model!r}")


def tune_and_fit(model, X, y, train, splits, cfg, seed, threads=1, quick=False) -> Fitted | None:
    """`tune` followed by `fit`. Returns a `Fitted`, or None if tuning found
    no setting that converged."""
    params, score = tune(model, X, y, train, splits, cfg, seed, threads, quick)
    if params is None:
        return None
    return fit(model, X, y, train, params, score, cfg, seed, threads, quick)
