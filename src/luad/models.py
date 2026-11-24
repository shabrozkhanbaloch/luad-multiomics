"""Two survival model families, tuned by inner cross-validation.

Higher risk always means worse prognosis. A fitted model is returned as a
`Fitted` with a `predict` function, so one fit can score many versions of the
test rows (for example with a block removed).
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
    comes from the training rows. NaN where the estimator is undefined."""
    tau = min(y_train["time"].max(), y_test["time"].max())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return float(concordance_index_ipcw(y_train, y_test, risk, tau=tau)[0])
        except (ValueError, ZeroDivisionError):
            return float("nan")


def harrell_c(y_test, risk) -> float:
    try:
        return float(concordance_index_censored(y_test["event"], y_test["time"], risk)[0])
    except (ValueError, ZeroDivisionError):
        return float("nan")


def _score(y_train, y_test, risk) -> float:
    s = uno_c(y_train, y_test, risk)
    return harrell_c(y_test, risk) if np.isnan(s) else s


def inner_splits(y, train: np.ndarray, n_folds: int, seed: int):
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return [(train[a], train[b]) for a, b in cv.split(train, y["event"][train].astype(int))]


# ---------------------------------------------------------------- fitted model
@dataclass
class Fitted:
    model: str
    params: dict
    inner_score: float                      # mean inner-CV C of the chosen setting
    predict: Callable[[np.ndarray], np.ndarray]
    train_risk: np.ndarray                  # risk of the training rows
    selected: np.ndarray | None = None      # columns with non-zero coefficient (coxnet)


def _scaler(X, train):
    mu, sd = X[train].mean(axis=0), X[train].std(axis=0)
    sd[sd < 1e-12] = 1.0
    return lambda A: ((A - mu) / sd).astype(np.float64)


# ---------------------------------------------------------------- penalised Cox
def _tune_coxnet(X, y, train, splits, cfg, quick):
    """One regularisation path per (l1_ratio, inner fold), scored at every
    alpha. The alpha grid comes from the outer-training rows so a given alpha
    means the same penalty in every inner fold."""
    g = cfg["models"]["coxnet"]
    ratios = g["l1_ratio"][:1] if quick else g["l1_ratio"]
    n_alphas = 15 if quick else int(g["n_alphas"])
    Xs = _scaler(X, train)(X)
    best, best_score = None, -np.inf
    for r in ratios:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                path = CoxnetSurvivalAnalysis(l1_ratio=r, n_alphas=n_alphas,
                                              alpha_min_ratio=0.01).fit(Xs[train], y[train])
            except (ValueError, ArithmeticError):
                continue
            alphas = np.asarray(path.alphas_, float)
            # The ends of the path are the null model and a barely penalised
            # fit that rarely converges when features outnumber patients.
            alphas = alphas[len(alphas) // 10: -1] if len(alphas) > 12 else alphas
            if not len(alphas):
                continue
            total, count = np.zeros(len(alphas)), np.zeros(len(alphas))
            for itr, ite in splits:
                # The scaler is refitted on the inner-training rows.
                Xi = _scaler(X, itr)(X)
                try:
                    m = CoxnetSurvivalAnalysis(l1_ratio=r, alphas=alphas).fit(Xi[itr], y[itr])
                except (ValueError, ArithmeticError):
                    continue
                for ai, a in enumerate(alphas):
                    risk = m.predict(Xi[ite], alpha=a)
                    if np.all(np.isfinite(risk)):
                        total[ai] += _score(y[itr], y[ite], risk)
                        count[ai] += 1
        if not (count > 0).any():
            continue
        mean = np.where(count > 0, total / np.maximum(count, 1), -np.inf)
        ai = int(np.argmax(mean))
        if mean[ai] > best_score:
            best_score = float(mean[ai])
            best = {"l1_ratio": r, "alpha": float(alphas[ai]), "alphas": alphas}
    return best, best_score


def _fit_coxnet(X, y, train, params, score) -> Fitted:
    scale = _scaler(X, train)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = CoxnetSurvivalAnalysis(l1_ratio=params["l1_ratio"],
                                   alphas=params["alphas"]).fit(scale(X[train]), y[train])
    a = params["alpha"]
    ai = int(np.argmin(np.abs(np.asarray(m.alphas_) - a)))
    predict = lambda A: m.predict(scale(A), alpha=a)  # noqa: E731
    return Fitted("coxnet", params, score, predict, predict(X[train]),
                  np.flatnonzero(m.coef_[:, ai] != 0))


# ---------------------------------------------------------------- random survival forest
def _rsf(params, n_trees, seed, threads):
    return RandomSurvivalForest(
        n_estimators=n_trees, max_features=params["max_features"],
        min_samples_leaf=params["min_samples_leaf"],
        min_samples_split=2 * params["min_samples_leaf"],
        n_jobs=threads, random_state=seed)


def _tune_rsf(X, y, train, splits, cfg, seed, threads, quick):
    g = cfg["models"]["rsf"]
    leaves = g["min_samples_leaf"][:1] if quick else g["min_samples_leaf"]
    n_trees = 50 if quick else int(g["n_estimators_tune"])
    best, best_score = None, -np.inf
    for leaf in leaves:
        params = {"max_features": g["max_features"], "min_samples_leaf": leaf}
        scores = []
        for itr, ite in splits:
            risk = _rsf(params, n_trees, seed, threads).fit(X[itr], y[itr]).predict(X[ite])
            scores.append(_score(y[itr], y[ite], risk))
        s = float(np.mean(scores))
        if s > best_score:
            best, best_score = params, s
    return best, best_score


def _fit_rsf(X, y, train, params, score, cfg, seed, threads, quick) -> Fitted:
    n_trees = 100 if quick else int(cfg["models"]["rsf"]["n_estimators_final"])
    m = _rsf(params, n_trees, seed, threads).fit(X[train], y[train])
    return Fitted("rsf", params, score, m.predict, m.predict(X[train]))


# ---------------------------------------------------------------- public
def tune(model: str, X: np.ndarray, y, train: np.ndarray, splits, cfg: dict,
         seed: int, threads: int = 1, quick: bool = False):
    """Best setting by inner CV: (params, mean inner C). params is None if
    nothing converged."""
    if model == "coxnet":
        return _tune_coxnet(X, y, train, splits, cfg, quick)
    if model == "rsf":
        return _tune_rsf(X, y, train, splits, cfg, seed, threads, quick)
    raise ValueError(f"unknown model {model!r}")


def fit(model: str, X: np.ndarray, y, train: np.ndarray, params: dict, score: float,
        cfg: dict, seed: int, threads: int = 1, quick: bool = False) -> Fitted:
    """Refit the chosen setting on all of `train`."""
    if model == "coxnet":
        return _fit_coxnet(X, y, train, params, score)
    return _fit_rsf(X, y, train, params, score, cfg, seed, threads, quick)


def tune_and_fit(model, X, y, train, splits, cfg, seed, threads=1, quick=False) -> Fitted | None:
    params, score = tune(model, X, y, train, splits, cfg, seed, threads, quick)
    if params is None:
        return None
    return fit(model, X, y, train, params, score, cfg, seed, threads, quick)
