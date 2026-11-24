"""The two integration methods that are more than concatenation.

Early fusion is concatenation and lives in transforms.concat / impute_missing.
Here: late fusion (one model per block, risks combined) and MOFA (shared
latent factors). Both can score a patient who lacks a block without imputing
that block's features.
"""
from __future__ import annotations

import contextlib
import hashlib
import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .models import Fitted


# ---------------------------------------------------------------- late fusion
def late_fusion(fits: dict[str, Fitted], risks: dict[str, np.ndarray],
                missing: list[str] | tuple = ()) -> np.ndarray:
    """Weighted mean of per-block risks over the blocks the patient has.

    Each block's risk is put on a common scale using the mean and SD of that
    model's risks on its own training rows. A block's weight is how far its
    inner-CV concordance sits above 0.5; a block no better than chance gets
    no weight. If every block is at chance the weights are equal.
    """
    use = [b for b in fits if b not in missing]
    if not use:
        raise ValueError("late fusion needs at least one available block")
    if len(use) == 1:
        return np.array(risks[use[0]], dtype=float)   # one block: its own risk, untouched
    w = np.array([max(fits[b].inner_score - 0.5, 0.0) for b in use])
    if w.sum() <= 0:
        w = np.ones(len(use))
    z = []
    for b in use:
        mu, sd = fits[b].train_risk.mean(), fits[b].train_risk.std()
        z.append((risks[b] - mu) / (sd if sd > 1e-12 else 1.0))
    return np.average(np.vstack(z), axis=0, weights=w)


# ---------------------------------------------------------------- MOFA
@dataclass
class Mofa:
    """A fitted factor model, kept as the few arrays needed to project new
    patients: per view the feature means, one scale, weights and noise
    precisions."""
    views: list[str]
    mean: dict[str, np.ndarray]
    scale: dict[str, float]
    W: dict[str, np.ndarray]        # features x factors
    tau: dict[str, np.ndarray]      # features
    active: np.ndarray              # factors kept
    r2: np.ndarray                  # views x factors, variance explained on training rows

    def _prep(self, view: str, X: np.ndarray) -> np.ndarray:
        return (X.astype(np.float64) - self.mean[view]) / self.scale[view]

    def project(self, X: dict[str, np.ndarray], missing: list[str] | tuple = ()) -> np.ndarray:
        """Posterior mean of the factors for new patients, given the fitted
        weights, using only the views that are present:

            Z = (I + sum_m W_m' T_m W_m)^-1  sum_m W_m' T_m x_m

        With no view present this is the prior mean, zero.
        """
        K = next(iter(self.W.values())).shape[1]
        n = next(iter(X.values())).shape[0]
        prec, rhs = np.eye(K), np.zeros((n, K))
        for v in self.views:
            if v in missing:
                continue
            WT = self.W[v] * self.tau[v][:, None]
            prec += self.W[v].T @ WT
            rhs += self._prep(v, X[v]) @ WT
        return np.linalg.solve(prec, rhs.T).T[:, self.active]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {"views": np.array(self.views), "active": self.active, "r2": self.r2}
        for v in self.views:
            arrays.update({f"mean__{v}": self.mean[v], f"scale__{v}": np.array(self.scale[v]),
                           f"W__{v}": self.W[v], f"tau__{v}": self.tau[v]})
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path: Path) -> "Mofa":
        z = np.load(path)
        views = [str(v) for v in z["views"]]
        return cls(views, {v: z[f"mean__{v}"] for v in views},
                   {v: float(z[f"scale__{v}"]) for v in views},
                   {v: z[f"W__{v}"] for v in views}, {v: z[f"tau__{v}"] for v in views},
                   z["active"], z["r2"])


def fit_mofa(X: dict[str, np.ndarray], cfg: dict, seed: int) -> Mofa:
    """Fit MOFA on the given rows (training rows only, when called in a fold).

    Each view is centred per feature and divided by one number, its overall
    SD, so a view with many features does not dominate through scale alone.
    All views use a Gaussian likelihood, including the 0/1 mutation view; that
    is an approximation, stated in the README.
    """
    from mofapy2.run.entry_point import entry_point

    m = cfg["mofa"]
    views = list(X)
    mean = {v: X[v].astype(np.float64).mean(axis=0) for v in views}
    scale = {v: float((X[v].astype(np.float64) - mean[v]).std()) or 1.0 for v in views}
    data = {v: (X[v].astype(np.float64) - mean[v]) / scale[v] for v in views}

    ent = entry_point()
    with contextlib.redirect_stdout(io.StringIO()):
        ent.set_data_options(scale_views=False)
        ent.set_data_matrix([[data[v]] for v in views], likelihoods=["gaussian"] * len(views))
        ent.set_model_options(factors=int(m["factors"]), spikeslab_weights=True, ard_weights=True)
        ent.set_train_options(iter=int(m["iterations"]), convergence_mode="fast",
                              seed=int(seed), verbose=False, quiet=True)
        ent.build()
        ent.run()
    exp = ent.model.getExpectations()
    W = {v: np.asarray(exp["W"][i]["E"], float) for i, v in enumerate(views)}
    tau = {v: np.asarray(exp["Tau"][i]["E"], float)[0] for i, v in enumerate(views)}

    model = Mofa(views, mean, scale, W, tau, np.arange(int(m["factors"])), np.zeros(0))
    # Variance explained per view and factor, from the same projection that
    # will be used on test patients, so "active" means active as used.
    Z = model.project(X)
    r2 = np.zeros((len(views), Z.shape[1]))
    for i, v in enumerate(views):
        total = float((data[v] ** 2).sum())
        for k in range(Z.shape[1]):
            resid = data[v] - np.outer(Z[:, k], W[v][:, k])
            r2[i, k] = 1.0 - float((resid ** 2).sum()) / total
    model.r2 = r2
    model.active = np.flatnonzero(r2.max(axis=0) >= float(m["min_factor_variance"]))
    if len(model.active) == 0:
        model.active = np.array([int(np.argmax(r2.max(axis=0)))])
    return model


def cached_mofa(X: dict[str, np.ndarray], cfg: dict, seed: int, cache_dir: Path | None) -> Mofa:
    """fit_mofa, cached on disk by the exact data and settings."""
    if cache_dir is None:
        return fit_mofa(X, cfg, seed)
    h = hashlib.sha1(repr((sorted(cfg["mofa"].items()), seed)).encode())
    for v in X:
        h.update(v.encode())
        h.update(np.ascontiguousarray(X[v]).tobytes())
    path = Path(cache_dir) / f"mofa_{h.hexdigest()[:16]}.npz"
    if path.exists():
        return Mofa.load(path)
    model = fit_mofa(X, cfg, seed)
    model.save(path)
    return model
