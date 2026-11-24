"""One outer fold of the cross-validation: every arm, every scenario, both models.

What this file is for
    Cross-validation answers "how well would this model do on patients it has
    never seen?". The patients are cut into parts called folds. Each fold takes
    its turn as the test set: the models are trained on all the other patients
    (the training rows) and give a risk to each patient of the fold (the test
    rows). This file does the work for ONE such fold. scripts/01_cv.py calls it
    for every fold and saves the risks; scripts/02_bootstrap.py turns the risks
    into concordance values with intervals.

Words used here (the README has the full list)
    block     one kind of data about a patient: clinical, rna (expression),
              mutation or methylation.
    arm       a way of using the blocks: one block alone ("rna"), a block added
              to clinical ("clinical+rna"), all four in one model ("early"),
              one model per block averaged ("late"), or MOFA factors ("mofa").
    scenario  what the test patient is missing: nothing ("full"), one omics
              block, or all of them. Models are always trained on complete data.
    risk      the number a model gives a patient; higher means a worse
              prognosis is predicted. Only the order of the risks matters.
    width     how many expression genes and methylation probes are kept (the
              most variable ones on the training rows). It is called k in the
              config file and is tuned like any other setting.
    inner CV  a second cross-validation inside the training rows only. It is
              used to choose settings, so the test rows are never used for
              choosing anything. Its score is the "inner C" (inner concordance).

The main functions
    outer_folds   cuts the cohort into folds.
    run_fold      trains and applies every arm for one fold. This is the one
                  the scripts call.
    key           the name under which a risk is stored, e.g. "late@no_rna".
    fold_seed     gives every fold its own random seed.

Why the arms share one loop
    The single-block and added-block arms are Part 1 of the analysis; early
    fusion, late fusion and MOFA under each scenario are Part 2. They are fitted
    in the same fold loop so that late fusion reuses exactly the single-block
    models that Part 1 reports, instead of fitting its own copies.
"""
from __future__ import annotations

import zlib
from pathlib import Path

import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold

from . import models as M
from .data import OMICS, Cohort
from .integrate import cached_mofa, late_fusion
from .transforms import build_blocks, concat, impute_missing

# Arms that are a plain concatenation of blocks: the blocks listed are put
# side by side into one table of features and one model is fitted on it.
CONCAT_ARMS = {
    "clinical": ["clinical"],
    "rna": ["rna"],
    "mutation": ["mutation"],
    "methylation": ["methylation"],
    "clinical+rna": ["clinical", "rna"],
    "clinical+mutation": ["clinical", "mutation"],
    "clinical+methylation": ["clinical", "methylation"],
    "early": ["clinical", "rna", "mutation", "methylation"],
}
# The four single-block arms. Late fusion averages the risks of these four.
SINGLE = ["clinical", "rna", "mutation", "methylation"]
# The blocks that have a width to tune. Clinical and mutation do not.
WIDTH_BLOCKS = {"rna", "methylation"}


def key(arm: str, scenario: str = "full") -> str:
    """The name a risk is stored under: the arm alone when nothing is missing,
    otherwise "arm@scenario", for example "late@no_rna"."""
    return arm if scenario == "full" else f"{arm}@{scenario}"


def outer_folds(c: Cohort, cfg: dict, seed: int, quick: bool = False):
    """Cut the cohort into the folds of the outer cross-validation.

    c      the cohort.
    cfg    the configuration; cfg["cv"] gives the number of folds and repeats.
    seed   random seed of the split, so the same folds come out on every run.
    quick  True for a fast trial run: one repeat instead of all of them.

    Returns a list of (fold_id, train_idx, test_idx). fold_id is text such as
    "r2f3" (repeat 2, fold 3). train_idx and test_idx are arrays of row numbers
    into the cohort. The split is stratified on event status: every fold gets
    about the same share of patients with an event.
    """
    repeats = 1 if quick else int(cfg["cv"]["outer_repeats"])
    cv = RepeatedStratifiedKFold(n_splits=int(cfg["cv"]["outer_folds"]),
                                 n_repeats=repeats, random_state=seed)
    n_folds = int(cfg["cv"]["outer_folds"])
    # The splitter only needs the number of patients and the label to stratify
    # on, so it is given a column of zeros in place of the data. It returns the
    # folds of repeat 1, then those of repeat 2, and so on; the position in
    # that list gives the repeat and fold numbers of the name.
    return [(f"r{i // n_folds + 1}f{i % n_folds + 1}", train_rows, test_rows)
            for i, (train_rows, test_rows)
            in enumerate(cv.split(np.zeros(c.n), c.y["event"].astype(int)))]


def fold_seed(seed: int, fold_id: str) -> int:
    """A seed of its own for each fold, worked out from the project seed and
    the fold's name.

    crc32 is a fixed checksum of the name. Python's built-in hash() is not used
    because it gives a different number in every Python session.
    """
    return seed + zlib.crc32(fold_id.encode()) % 100_000


def _tune_widths(model, blocks_by_k, which, y, tr, splits, cfg, seed, threads, quick):
    """Tune one arm at each candidate width and keep the width that does best.

    model        "coxnet" (penalised Cox) or "rsf" (random survival forest).
    blocks_by_k  {width: {block name: Block}}, the four blocks built at each width.
    which        the block names this arm uses, e.g. ["clinical", "rna"].
    y            outcome of every patient (structured array: event, time).
    tr           row numbers of the training patients.
    splits       the inner cross-validation splits of the training rows.
    cfg, seed, threads, quick   passed on to the model tuning.

    Returns (params, score, k, X, owner) for the best width, or None if the
    model could not be tuned at any width. params are the chosen settings,
    score is their inner C, k the width, X the table of features (all patients
    x features) and owner the block each column of X belongs to.
    """
    # An arm with no expression or methylation looks the same at every width,
    # so it is tuned once, at the first width.
    ks = list(blocks_by_k) if WIDTH_BLOCKS & set(which) else list(blocks_by_k)[:1]
    best = None
    for k in ks:
        X, owner = concat(blocks_by_k[k], which)
        params, score = M.tune(model, X, y, tr, splits, cfg, seed, threads, quick)
        # Strictly better only: on a tie the earlier width in the list is kept.
        if params is not None and (best is None or score > best[1]):
            best = (params, score, k, X, owner)
    return best


def run_fold(c: Cohort, tr: np.ndarray, te: np.ndarray, fold_id: str, cfg: dict,
             seed: int, quick: bool = False, threads: int = 1,
             cache_dir: Path | None = None, arms: list[str] | None = None) -> dict:
    """Train every arm on the training rows of one fold and give each test
    patient a risk, for both models and under every scenario.

    c          the cohort (all patients, all four blocks).
    tr         row numbers of the training patients of this fold.
    te         row numbers of the test patients of this fold.
    fold_id    name of the fold, e.g. "r1f3".
    cfg        the configuration (config/config.yaml as a dict).
    seed       the project seed; the fold's own seed is worked out from it.
    quick      True for a fast trial run with small settings.
    threads    processor threads one model may use.
    cache_dir  folder where fitted MOFA models are kept between runs, or None
               to fit MOFA every time.
    arms       arm names to run, or None for all of them. Asking for "late"
               also runs the four single-block arms, which it is built from.

    Returns a dict with these keys:
        "fold"   fold_id.
        "train"  tr, as given.
        "test"   te, as given.
        "risk"   {model: {"arm" or "arm@scenario": risks of the test patients}},
                 each an array with one value per row of te, in the same order.
        "info"   a list with one dict per model and arm: the inner C, the
                 number of features, the chosen width "k" (for "mofa" this
                 column holds the number of factors kept) and, for penalised
                 Cox, the names of the features the model kept ("selected").
    An arm whose model could not be tuned is left out of "risk" and "info".
    """
    train_rows, test_rows = tr, te
    y = c.y
    # From here on `seed` is this fold's own seed.
    seed = fold_seed(seed, fold_id)
    # The inner splits are made once and shared by every arm and both models,
    # so all settings are chosen on the same inner folds.
    splits = M.inner_splits(y, train_rows, int(cfg["cv"]["inner_folds"]), seed)

    # Build the four blocks at each candidate width. Which genes and probes are
    # kept, and how missing clinical values are filled in, is learned from the
    # training rows only.
    widths = cfg["features"]["k_grid"][:1] if quick else cfg["features"]["k_grid"]
    blocks_by_width = {int(k): build_blocks(c, train_rows, cfg, seed, int(k)) for k in widths}
    scenarios = cfg["scenarios"]          # {scenario name: list of missing blocks}
    wanted = set(arms) if arms else set(CONCAT_ARMS) | {"late", "mofa"}
    if "late" in wanted:
        wanted |= set(SINGLE)

    # MOFA is fitted once per fold, on training rows, and shared by both models.
    if "mofa" in wanted:
        # MOFA gets its own, wider blocks (cfg["mofa"]["top_k"] features).
        wide = build_blocks(c, train_rows, cfg, seed, int(cfg["mofa"]["top_k"]))
        views = {v: wide[v].X for v in OMICS}
        mofa = cached_mofa({v: views[v][train_rows] for v in OMICS}, cfg, seed, cache_dir)
        clinical = wide["clinical"].X
        # Training rows are projected the same way test rows will be.
        X_mofa = np.hstack([clinical, mofa.project(views)]).astype(np.float32)

    risk, info = {}, []
    for model in M.MODELS:
        # test_risk: {arm or arm@scenario: risks of the test patients}.
        # fitted_arms: {arm: fitted model}, kept for late fusion below.
        test_risk, fitted_arms = {}, {}

        # ---- The arms that are one model on blocks placed side by side.
        for arm, which in CONCAT_ARMS.items():
            if arm not in wanted:
                continue
            # Choose the settings and the width by inner CV, then refit the
            # chosen settings on all the training rows.
            best = _tune_widths(model, blocks_by_width, which, y, train_rows, splits, cfg, seed,
                                threads, quick)
            if best is None:
                continue
            params, score, k, X, owner = best
            fitted = M.fit(model, X, y, train_rows, params, score, cfg, seed, threads, quick)
            fitted_arms[arm] = fitted
            test_risk[key(arm)] = fitted.predict(X[test_rows])

            # Record what was chosen, for the tables and for the biology part.
            names = [name for block in which for name in blocks_by_width[k][block].names]
            info.append({"fold": fold_id, "model": model, "arm": arm, "inner_c": score,
                         "n_features": X.shape[1],
                         "k": k if WIDTH_BLOCKS & set(which) else None,
                         "selected": ";".join(names[i] for i in fitted.selected)
                         if fitted.selected is not None else None})
            if arm == "early":
                # Early fusion is one model that needs every column. When a
                # test patient lacks a block, the same fitted model is used
                # and a missing block's columns take their training means.
                for scenario, missing in scenarios.items():
                    if missing:
                        test_risk[key(arm, scenario)] = fitted.predict(
                            impute_missing(X, owner, train_rows, missing)[test_rows])

        # ---- Late fusion: average the risks of the four single-block models
        # fitted above. A missing block's model is simply left out.
        if "late" in wanted and all(block in fitted_arms for block in SINGLE):
            single = {block: fitted_arms[block] for block in SINGLE}
            risks = {block: test_risk[key(block)] for block in SINGLE}
            for scenario, missing in scenarios.items():
                test_risk[key("late", scenario)] = late_fusion(single, risks, missing)

        # ---- MOFA: one model on the clinical covariates plus the factors.
        if "mofa" in wanted:
            fitted = M.tune_and_fit(model, X_mofa, y, train_rows, splits, cfg, seed, threads, quick)
            if fitted is not None:
                info.append({"fold": fold_id, "model": model, "arm": "mofa",
                             "inner_c": fitted.inner_score, "n_features": X_mofa.shape[1],
                             "k": len(mofa.active), "selected": None})
                test_views = {v: views[v][test_rows] for v in OMICS}
                # The factors of a test patient are computed from the blocks
                # the patient has; the fitted model is the same in every scenario.
                for scenario, missing in scenarios.items():
                    Z = mofa.project(test_views, missing)
                    test_risk[key("mofa", scenario)] = fitted.predict(
                        np.hstack([clinical[test_rows], Z]).astype(np.float32))
        risk[model] = test_risk
    return {"fold": fold_id, "train": tr, "test": te, "risk": risk, "info": info}
