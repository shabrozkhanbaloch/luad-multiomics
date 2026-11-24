"""The two integration methods that are more than concatenation.

What this file is for
    "Integration" means combining the four data blocks into one prediction.
    There are three ways in this project:

    early fusion   put all the columns side by side and fit one model. That
                   is concatenation and lives in transforms.concat /
                   impute_missing, not here.
    late fusion    fit one model per block and combine their risks.
    MOFA           summarise the three omics blocks by a few shared numbers
                   per patient (factors) and fit one model on those plus the
                   clinical covariates.

    Here: late fusion and MOFA. Both can score a patient who lacks a block
    without imputing that block's features.

Factors
    MOFA (Multi-Omics Factor Analysis) assumes that a small number of hidden
    quantities, the factors, drive the variation in all the omics blocks at
    once. Each patient has a value on each factor. Each feature (a gene, a
    probe) has a weight on each factor, saying how strongly the factor moves
    it. Thousands of features are thus replaced by a handful of factor
    values per patient. MOFA calls a data block a "view".

Main functions
    late_fusion    combine per-block risks into one risk
    fit_mofa       learn the factors from training patients
    Mofa.project   compute the factor values of any patients, new ones
                   included, from whichever views they have
    cached_mofa    fit_mofa, but reuse a result saved on disk when the same
                   data and settings were fitted before (MOFA is the slowest fit)
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

    fits
        Block name -> the model fitted on that block alone.
    risks
        Block name -> that model's risks for the patients to score, one
        number per patient. Every block must cover the same patients in the
        same order.
    missing
        Names of the blocks these patients do not have. Their models are
        left out.

    Returns one combined risk per patient, higher meaning worse.
    """
    available = [block for block in fits if block not in missing]
    if not available:
        raise ValueError("late fusion needs at least one available block")
    if len(available) == 1:
        return np.array(risks[available[0]], dtype=float)   # one block: its own risk, untouched

    # A block is trusted in proportion to how well it did in inner
    # cross-validation, which used training rows only.
    weights = np.array([max(fits[block].inner_score - 0.5, 0.0) for block in available])
    if weights.sum() <= 0:
        weights = np.ones(len(available))

    # Risks from different models are on different scales (a Cox risk and a
    # forest risk are not comparable numbers), so each is standardised
    # before averaging. The mean and SD come from training rows.
    standardised = []
    for block in available:
        mu, sd = fits[block].train_risk.mean(), fits[block].train_risk.std()
        standardised.append((risks[block] - mu) / (sd if sd > 1e-12 else 1.0))
    return np.average(np.vstack(standardised), axis=0, weights=weights)


# ---------------------------------------------------------------- MOFA
@dataclass
class Mofa:
    """A fitted factor model, kept as the few arrays needed to project new
    patients: per view the feature means, one scale, weights and noise
    precisions.

    The dictionaries are keyed by view name. "Precision" is one over the
    noise variance of a feature: a feature the factors explain well has a
    high precision and counts for more when a patient is projected.
    """
    views: list[str]                # view names, in the order they were fitted
    mean: dict[str, np.ndarray]     # per view: the training mean of each feature
    scale: dict[str, float]         # per view: one number the whole view is divided by
    W: dict[str, np.ndarray]        # features x factors
    tau: dict[str, np.ndarray]      # features
    active: np.ndarray              # factors kept
    r2: np.ndarray                  # views x factors, variance explained on training rows

    def _prep(self, view: str, X: np.ndarray) -> np.ndarray:
        """Centre and scale one view exactly as was done before fitting.

        X is patients x features for that view. Returns float64 of the same shape.
        """
        return (X.astype(np.float64) - self.mean[view]) / self.scale[view]

    def project(self, X: dict[str, np.ndarray], missing: list[str] | tuple = ()) -> np.ndarray:
        """Posterior mean of the factors for new patients, given the fitted
        weights, using only the views that are present:

            Z = (I + sum_m W_m' T_m W_m)^-1  sum_m W_m' T_m x_m

        With no view present this is the prior mean, zero.

        In words: this is the best estimate of each patient's factor values
        from their data. In the formula, m runs over the views the patient
        has, W_m is the weights of view m, T_m its precisions (on a
        diagonal) and x_m the patient's centred and scaled data. The
        identity matrix I is the prior belief that factors are near zero; it
        pulls the estimate towards zero when there is little data.

        X
            View name -> patients x features, with the same features, in the
            same order, as at fitting. Views listed in `missing` are not read.
        missing
            Names of the views these patients do not have.

        Returns patients x active factors.
        """
        K = next(iter(self.W.values())).shape[1]      # number of factors fitted
        n = next(iter(X.values())).shape[0]           # number of patients

        # Build the two parts of the formula by adding one view at a time:
        # `prec` is the matrix in brackets, `rhs` the sum on the right.
        prec, rhs = np.eye(K), np.zeros((n, K))
        for view in self.views:
            # A missing view simply adds nothing to either sum.
            if view in missing:
                continue
            weights_times_precision = self.W[view] * self.tau[view][:, None]
            prec += self.W[view].T @ weights_times_precision
            rhs += self._prep(view, X[view]) @ weights_times_precision

        # Solve for all patients at once, then keep only the active factors.
        return np.linalg.solve(prec, rhs.T).T[:, self.active]

    def save(self, path: Path) -> None:
        """Write the model to one compressed .npz file, creating the folder if needed."""
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {"views": np.array(self.views), "active": self.active, "r2": self.r2}
        # An .npz file is a flat set of named arrays, so the per-view
        # dictionaries are stored under names such as "W__rna".
        for view in self.views:
            arrays.update({f"mean__{view}": self.mean[view], f"scale__{view}": np.array(self.scale[view]),
                           f"W__{view}": self.W[view], f"tau__{view}": self.tau[view]})
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path: Path) -> "Mofa":
        """Read back a model written by `save`."""
        stored = np.load(path)
        views = [str(view) for view in stored["views"]]
        return cls(views, {view: stored[f"mean__{view}"] for view in views},
                   {view: float(stored[f"scale__{view}"]) for view in views},
                   {view: stored[f"W__{view}"] for view in views},
                   {view: stored[f"tau__{view}"] for view in views},
                   stored["active"], stored["r2"])


def fit_mofa(X: dict[str, np.ndarray], cfg: dict, seed: int) -> Mofa:
    """Fit MOFA on the given rows (training rows only, when called in a fold).

    Each view is centred per feature and divided by one number, its overall
    SD, so a view with many features does not dominate through scale alone.
    All views use a Gaussian likelihood, including the 0/1 mutation view; that
    is an approximation, stated in the README.

    X
        View name -> patients x features. Every view has the same patients
        in the same order.
    cfg
        The configuration. Under `mofa`: `factors` (how many to fit),
        `iterations`, and `min_factor_variance` (how much variance a factor
        must explain to be kept).
    seed
        Fixes the random starting point of the fit.

    Returns a `Mofa`. Its `active` lists the factors that are kept.
    """
    from mofapy2.run.entry_point import entry_point

    settings = cfg["mofa"]
    views = list(X)

    # Centre each feature and put each view on one scale. The means and
    # scales are kept in the model so new patients get the same treatment.
    mean = {view: X[view].astype(np.float64).mean(axis=0) for view in views}
    scale = {view: float((X[view].astype(np.float64) - mean[view]).std()) or 1.0 for view in views}
    data = {view: (X[view].astype(np.float64) - mean[view]) / scale[view] for view in views}

    # Run the MOFA library. The progress text it prints while it is set up
    # and trained is caught and thrown away, so the pipeline's own output
    # stays readable.
    mofa_run = entry_point()
    with contextlib.redirect_stdout(io.StringIO()):
        # The views were scaled above, so the library must not scale them again.
        mofa_run.set_data_options(scale_views=False)
        mofa_run.set_data_matrix([[data[view]] for view in views], likelihoods=["gaussian"] * len(views))
        mofa_run.set_model_options(factors=int(settings["factors"]), spikeslab_weights=True,
                                   ard_weights=True)
        mofa_run.set_train_options(iter=int(settings["iterations"]), convergence_mode="fast",
                                   seed=int(seed), verbose=False, quiet=True)
        mofa_run.build()
        mofa_run.run()

    # Take out what projection needs: the weights and the noise precisions.
    # The library's own factor values for the training patients are not used.
    expectations = mofa_run.model.getExpectations()
    W = {view: np.asarray(expectations["W"][i]["E"], float) for i, view in enumerate(views)}
    tau = {view: np.asarray(expectations["Tau"][i]["E"], float)[0] for i, view in enumerate(views)}

    # To begin with every factor counts as active; the real choice is made below.
    model = Mofa(views, mean, scale, W, tau, np.arange(int(settings["factors"])), np.zeros(0))

    # Variance explained per view and factor, from the same projection that
    # will be used on test patients, so "active" means active as used.
    Z = model.project(X)
    r2 = np.zeros((len(views), Z.shape[1]))
    for i, view in enumerate(views):
        total = float((data[view] ** 2).sum())
        for k in range(Z.shape[1]):
            # What is left of the view after removing this one factor's part.
            resid = data[view] - np.outer(Z[:, k], W[view][:, k])
            r2[i, k] = 1.0 - float((resid ** 2).sum()) / total
    model.r2 = r2

    # Keep a factor if it explains enough of at least one view. Factors
    # that explain almost nothing are noise and would only add columns.
    model.active = np.flatnonzero(r2.max(axis=0) >= float(settings["min_factor_variance"]))
    # Never return a model with no factor: keep the single best one.
    if len(model.active) == 0:
        model.active = np.array([int(np.argmax(r2.max(axis=0)))])
    return model


def cached_mofa(X: dict[str, np.ndarray], cfg: dict, seed: int, cache_dir: Path | None) -> Mofa:
    """fit_mofa, cached on disk by the exact data and settings.

    X, cfg, seed
        As in `fit_mofa`.
    cache_dir
        The folder for saved fits. None switches the cache off.

    Returns a `Mofa`, read from disk if this exact fit was done before.
    """
    if cache_dir is None:
        return fit_mofa(X, cfg, seed)

    # The file name is a fingerprint (hash) of everything the fit depends
    # on: the MOFA settings, the seed, and every number in every view. If
    # any of them changes, the name changes and the model is fitted afresh.
    fingerprint = hashlib.sha1(repr((sorted(cfg["mofa"].items()), seed)).encode())
    for view in X:
        fingerprint.update(view.encode())
        fingerprint.update(np.ascontiguousarray(X[view]).tobytes())
    path = Path(cache_dir) / f"mofa_{fingerprint.hexdigest()[:16]}.npz"

    if path.exists():
        return Mofa.load(path)
    model = fit_mofa(X, cfg, seed)
    model.save(path)
    return model
