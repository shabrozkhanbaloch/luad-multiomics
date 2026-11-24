"""Shared by the numbered scripts: threads, paths, saving and loading predictions.

What it is for
    Every numbered script starts with `from _common import ...`. That one line
    does three things before any heavy library is loaded:

    1. limits each worker process to one numerical thread,
    2. lowers the priority of the run, so the computer stays usable,
    3. makes the `luad` library in `src/` importable.

    It also holds the few helpers that more than one script needs.

Why the import order matters
    The thread limits only work if they are set before numpy is first
    imported. This is why every script imports `_common` before numpy, and why
    the imports below come after the settings instead of at the top.

What it provides
    load_config        read config/config.yaml (passed on from the library)
    n_jobs             how many worker processes to use
    out_dirs           the folders for tables and figures
    save_predictions   write the risks of step 1 to one file
    load_predictions   read them back for step 2
    md_table           a table as markdown text
"""
from __future__ import annotations

import os

# One BLAS thread per worker; the parallelism is across folds. Must be set
# before numpy is imported. (BLAS is the linear-algebra library under numpy.
# Without this, every worker would start one thread per processor core and
# the workers would slow each other down.)
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

# Run at the lowest priority, so a long analysis never makes the desktop
# (browser, editor, Cytoscape) unresponsive. Worker processes inherit it.
# Some systems do not allow a program to change its priority; the run then
# simply continues at normal priority.
try:
    os.nice(19)
except OSError:
    pass

import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# The library lives in src/luad and is not installed as a package, so its
# folder is added to the places Python searches for imports.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from luad import load_config  # noqa: E402,F401


def n_jobs(requested: int | None = None) -> int:
    """Workers to use. By default a quarter of the CPU threads are left free
    for the desktop; pass --jobs to use more or fewer.

    Arguments
        requested   the value of the --jobs flag, or None if it was not given

    Returns
        The number of worker processes. With no request: all CPU threads minus
        a quarter of them (minus at least 2), and never fewer than 1.
    """
    cpus = os.cpu_count() or 2
    return requested or max(1, cpus - max(2, cpus // 4))


def out_dirs(cfg: dict) -> tuple[Path, Path]:
    """The two output folders, created if they do not exist yet.

    Arguments
        cfg   the plan, as returned by load_config()

    Returns
        (tables folder, figures folder), normally results/tables and
        results/figures.
    """
    tables, figures = Path(cfg["paths"]["tables"]), Path(cfg["paths"]["figures"])
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    return tables, figures


def save_predictions(results: list[dict], path: Path) -> None:
    """Long table: fold, role (train/test), patient index, and for test rows
    one risk per model and arm.

    Arguments
        results   one dict per outer fold, as returned by luad.cv.run_fold:
                  {"fold": name, "train": row numbers, "test": row numbers,
                   "risk": {model: {arm: one risk per test patient}}}
        path      the parquet file to write

    Returns
        Nothing. The file has the columns fold, model, arm, patient, risk.
    """
    rows = []
    for fold_result in results:
        # The training patients of a fold are stored too, under the arm name
        # "__train__" and with no risk. Step 2 needs them: Uno's C weights each
        # test patient by the censoring pattern of the training patients.
        rows.append(pd.DataFrame({"fold": fold_result["fold"], "model": "", "arm": "__train__",
                                  "patient": fold_result["train"], "risk": np.nan}))
        for model, arms in fold_result["risk"].items():
            for arm, risk in arms.items():
                rows.append(pd.DataFrame({"fold": fold_result["fold"], "model": model, "arm": arm,
                                          "patient": fold_result["test"], "risk": risk}))
    pd.concat(rows, ignore_index=True).to_parquet(path, index=False)


def load_predictions(path: Path, model: str) -> list[dict]:
    """Back into the fold dicts the bootstrap takes, for one model.

    Arguments
        path    the parquet file written by save_predictions
        model   which model's risks to read, for example "coxnet" or "rsf"

    Returns
        One dict per fold, in the order the folds were saved:
        {"fold": name, "train": row numbers, "test": row numbers,
         "risk": {arm: one risk per test patient}}.
        A fold with no risks for this model is left out.
    """
    predictions = pd.read_parquet(path)
    folds = []
    for fold_id, fold_rows in predictions.groupby("fold", sort=False):
        train = fold_rows.loc[fold_rows["arm"] == "__train__", "patient"].to_numpy()
        model_rows = fold_rows[fold_rows["model"] == model]
        risk, test = {}, None
        for arm, arm_rows in model_rows.groupby("arm", sort=False):
            # Every arm of a fold must list the same test patients in the same
            # order. Otherwise two arms could not be compared patient by patient.
            test = arm_rows["patient"].to_numpy() if test is None else test
            assert np.array_equal(test, arm_rows["patient"].to_numpy()), "arms disagree on test rows"
            risk[arm] = arm_rows["risk"].to_numpy()
        if test is not None:
            folds.append({"fold": fold_id, "train": train, "test": test, "risk": risk})
    return folds


def md_table(df: pd.DataFrame, floatfmt: str = ".3f") -> str:
    """A table as markdown text, without the row numbers.

    Arguments
        df         the table
        floatfmt   how to print decimal numbers; ".3f" is three decimal places

    Returns
        The markdown text of the table, as one string.
    """
    return df.to_markdown(index=False, floatfmt=floatfmt)
