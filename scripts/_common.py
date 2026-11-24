"""Shared by the numbered scripts: threads, paths, saving and loading predictions."""
from __future__ import annotations

import os

# One BLAS thread per worker; the parallelism is across folds. Must be set
# before numpy is imported.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

# Run at the lowest priority, so a long analysis never makes the desktop
# (browser, editor, Cytoscape) unresponsive. Worker processes inherit it.
try:
    os.nice(19)
except OSError:
    pass

import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from luad import load_config  # noqa: E402,F401


def n_jobs(requested: int | None = None) -> int:
    """Workers to use. By default a quarter of the CPU threads are left free
    for the desktop; pass --jobs to use more or fewer."""
    cpus = os.cpu_count() or 2
    return requested or max(1, cpus - max(2, cpus // 4))


def out_dirs(cfg: dict) -> tuple[Path, Path]:
    tables, figures = Path(cfg["paths"]["tables"]), Path(cfg["paths"]["figures"])
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    return tables, figures


def save_predictions(results: list[dict], path: Path) -> None:
    """Long table: fold, role (train/test), patient index, and for test rows
    one risk per model and arm."""
    rows = []
    for r in results:
        rows.append(pd.DataFrame({"fold": r["fold"], "model": "", "arm": "__train__",
                                  "patient": r["train"], "risk": np.nan}))
        for model, arms in r["risk"].items():
            for arm, risk in arms.items():
                rows.append(pd.DataFrame({"fold": r["fold"], "model": model, "arm": arm,
                                          "patient": r["test"], "risk": risk}))
    pd.concat(rows, ignore_index=True).to_parquet(path, index=False)


def load_predictions(path: Path, model: str) -> list[dict]:
    """Back into the fold dicts the bootstrap takes, for one model."""
    df = pd.read_parquet(path)
    folds = []
    for fid, g in df.groupby("fold", sort=False):
        train = g.loc[g["arm"] == "__train__", "patient"].to_numpy()
        gm = g[g["model"] == model]
        risk, test = {}, None
        for arm, ga in gm.groupby("arm", sort=False):
            test = ga["patient"].to_numpy() if test is None else test
            assert np.array_equal(test, ga["patient"].to_numpy()), "arms disagree on test rows"
            risk[arm] = ga["risk"].to_numpy()
        if test is not None:
            folds.append({"fold": fid, "train": train, "test": test, "risk": risk})
    return folds


def md_table(df: pd.DataFrame, floatfmt: str = ".3f") -> str:
    return df.to_markdown(index=False, floatfmt=floatfmt)
