"""One outer fold: every arm, every scenario, both models.

An arm is a way of using the blocks. A scenario is what the test patient is
missing. The single-block and added-block arms are Part 1; early fusion, late
fusion and MOFA under each scenario are Part 2. They share one fold loop so
late fusion reuses exactly the single-block models Part 1 reports.
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

# Arms that are a plain concatenation of blocks.
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
SINGLE = ["clinical", "rna", "mutation", "methylation"]
WIDTH_BLOCKS = {"rna", "methylation"}


def key(arm: str, scenario: str = "full") -> str:
    return arm if scenario == "full" else f"{arm}@{scenario}"


def outer_folds(c: Cohort, cfg: dict, seed: int, quick: bool = False):
    """[(fold_id, train_idx, test_idx)], stratified on event status."""
    repeats = 1 if quick else int(cfg["cv"]["outer_repeats"])
    cv = RepeatedStratifiedKFold(n_splits=int(cfg["cv"]["outer_folds"]),
                                 n_repeats=repeats, random_state=seed)
    n_folds = int(cfg["cv"]["outer_folds"])
    return [(f"r{i // n_folds + 1}f{i % n_folds + 1}", tr, te)
            for i, (tr, te) in enumerate(cv.split(np.zeros(c.n), c.y["event"].astype(int)))]


def fold_seed(seed: int, fold_id: str) -> int:
    return seed + zlib.crc32(fold_id.encode()) % 100_000


def _tune_widths(model, blocks_by_k, which, y, tr, splits, cfg, seed, threads, quick):
    """Tune the model at each candidate width; keep the best by inner CV."""
    ks = list(blocks_by_k) if WIDTH_BLOCKS & set(which) else list(blocks_by_k)[:1]
    best = None
    for k in ks:
        X, owner = concat(blocks_by_k[k], which)
        params, score = M.tune(model, X, y, tr, splits, cfg, seed, threads, quick)
        if params is not None and (best is None or score > best[1]):
            best = (params, score, k, X, owner)
    return best


def run_fold(c: Cohort, tr: np.ndarray, te: np.ndarray, fold_id: str, cfg: dict,
             seed: int, quick: bool = False, threads: int = 1,
             cache_dir: Path | None = None, arms: list[str] | None = None) -> dict:
    """Returns {"fold", "train", "test", "risk": {model: {arm@scenario: risk on test}},
    "info": [one row per model and arm]}."""
    y = c.y
    seed = fold_seed(seed, fold_id)
    splits = M.inner_splits(y, tr, int(cfg["cv"]["inner_folds"]), seed)
    widths = cfg["features"]["k_grid"][:1] if quick else cfg["features"]["k_grid"]
    by_k = {int(k): build_blocks(c, tr, cfg, seed, int(k)) for k in widths}
    scenarios = cfg["scenarios"]
    want = set(arms) if arms else set(CONCAT_ARMS) | {"late", "mofa"}
    if "late" in want:
        want |= set(SINGLE)

    # MOFA is fitted once per fold, on training rows, and shared by both models.
    if "mofa" in want:
        wide = build_blocks(c, tr, cfg, seed, int(cfg["mofa"]["top_k"]))
        views = {v: wide[v].X for v in OMICS}
        mofa = cached_mofa({v: views[v][tr] for v in OMICS}, cfg, seed, cache_dir)
        clin = wide["clinical"].X
        # Training rows are projected the same way test rows will be.
        X_mofa = np.hstack([clin, mofa.project(views)]).astype(np.float32)

    risk, info = {}, []
    for model in M.MODELS:
        out, fits = {}, {}
        for arm, which in CONCAT_ARMS.items():
            if arm not in want:
                continue
            best = _tune_widths(model, by_k, which, y, tr, splits, cfg, seed, threads, quick)
            if best is None:
                continue
            params, score, k, X, owner = best
            f = M.fit(model, X, y, tr, params, score, cfg, seed, threads, quick)
            fits[arm] = f
            out[key(arm)] = f.predict(X[te])
            names = [n for b in which for n in by_k[k][b].names]
            info.append({"fold": fold_id, "model": model, "arm": arm, "inner_c": score,
                         "n_features": X.shape[1],
                         "k": k if WIDTH_BLOCKS & set(which) else None,
                         "selected": ";".join(names[i] for i in f.selected)
                         if f.selected is not None else None})
            if arm == "early":
                # A missing block's columns take their training means.
                for sc, missing in scenarios.items():
                    if missing:
                        out[key(arm, sc)] = f.predict(impute_missing(X, owner, tr, missing)[te])

        if "late" in want and all(b in fits for b in SINGLE):
            single = {b: fits[b] for b in SINGLE}
            risks = {b: out[key(b)] for b in SINGLE}
            for sc, missing in scenarios.items():
                out[key("late", sc)] = late_fusion(single, risks, missing)

        if "mofa" in want:
            f = M.tune_and_fit(model, X_mofa, y, tr, splits, cfg, seed, threads, quick)
            if f is not None:
                info.append({"fold": fold_id, "model": model, "arm": "mofa",
                             "inner_c": f.inner_score, "n_features": X_mofa.shape[1],
                             "k": len(mofa.active), "selected": None})
                test_views = {v: views[v][te] for v in OMICS}
                for sc, missing in scenarios.items():
                    Z = mofa.project(test_views, missing)
                    out[key("mofa", sc)] = f.predict(np.hstack([clin[te], Z]).astype(np.float32))
        risk[model] = out
    return {"fold": fold_id, "train": tr, "test": te, "risk": risk, "info": info}
