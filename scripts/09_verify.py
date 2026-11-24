#!/usr/bin/env python
"""Step 9. Check a run against the published numbers.

    python scripts/09_verify.py                    # compare, print PASS or the differences
    python scripts/09_verify.py --write-reference  # maintainers only: save the current
                                                   # tables as the new reference

What it is for
    After `make all`, this answers one question without any reading of tables:
    "did I get the same results as the authors?"

What it reads
    results/tables/         the tables your run just wrote
    tests/reference_tables/ the same tables from the published run

What it prints
    One line per table: PASS, or the largest difference and where it is.
    The command ends with an error if any table differs, so it can be used
    in a script.

How long it takes
    About a second. Nothing is fitted.

How strict it is
    Numbers must agree to 6 decimal places and text must be identical. With
    the package versions in requirements.txt the tables come out identical to
    the last digit; the small allowance is for other processors and operating
    systems, where the last digits of a floating-point sum can differ.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from _common import load_config, out_dirs

import numpy as np
import pandas as pd

from luad import ROOT

REFERENCE_DIR = ROOT / "tests" / "reference_tables"

# Every table that carries a reported number. The large files of per-patient
# predictions are left out: the tables below are computed from them, so a
# difference there would show up here.
TABLES = [
    "cohort_flow.csv",               # how 509 patients become 440
    "cohort_characteristics.csv",    # baseline table
    "cv_pfi_summary.csv",            # mean concordance per arm, primary endpoint
    "cv_os_summary.csv",             # the same, overall survival
    "arms_pfi.csv",                  # concordance with intervals
    "arms_os.csv",
    "contrasts_pfi.csv",             # the 34 pre-declared comparisons
    "contrasts_os.csv",
    "external_arms.csv",             # the three GEO cohorts
    "external_contrasts.csv",
    "biology_factors.csv",           # MOFA factors
    "biology_overlap.csv",           # shared genes between lists
    "biology_modules.csv",           # network modules
    "biology_enrichment.csv",        # what each module is enriched for
    "biology_network_links.csv",     # interactions between lists
    "biology_network_centrality.csv",
    "cv_pfi_permuted_summary.csv",   # shuffled-outcome check
    "checks.csv",                    # the robustness checks
    "width_sweep.csv",
    "survival_groups_pfi.csv",       # added after the first run: risk groups (step 10)
    "survival_groups_os.csv",
    "survival_curves_pfi.csv",
    "survival_curves_os.csv",
    "biology_factor_scores.csv",     # each patient's value on each MOFA factor (step 4)
    "biology_factor_associations.csv",  # added after the first run: what factor 2 follows (step 11)
    "biology_factor_outcome.csv",
]

TOLERANCE = 1e-6


def compare(mine: pd.DataFrame, reference: pd.DataFrame) -> str | None:
    """None if the two tables agree, otherwise one sentence saying how they differ."""
    if list(mine.columns) != list(reference.columns):
        return "the columns are different"
    if len(mine) != len(reference):
        return f"{len(mine)} rows, the reference has {len(reference)}"

    for column in reference.columns:
        if pd.api.types.is_numeric_dtype(reference[column]) and pd.api.types.is_numeric_dtype(mine[column]):
            a = mine[column].to_numpy(float)
            b = reference[column].to_numpy(float)
            # A missing value must be missing in both.
            if (np.isnan(a) != np.isnan(b)).any():
                return f"column `{column}`: a value is missing in one table and present in the other"
            gap = np.nanmax(np.abs(a - b)) if len(a) else 0.0
            if gap > TOLERANCE:
                row = int(np.nanargmax(np.abs(a - b)))
                return (f"column `{column}`, row {row + 1}: {a[row]:.6g} here, "
                        f"{b[row]:.6g} in the reference (difference {gap:.2g})")
        else:
            # An empty cell is written as "" so that two empty cells count as equal.
            a = mine[column].astype(object).where(mine[column].notna(), "").astype(str).to_numpy()
            b = reference[column].astype(object).where(reference[column].notna(), "").astype(str).to_numpy()
            if (a != b).any():
                row = int(np.flatnonzero(a != b)[0])
                return f"column `{column}`, row {row + 1}: `{a[row]}` here, `{b[row]}` in the reference"
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--write-reference", action="store_true",
                        help="save the current tables as the reference (maintainers only)")
    args = parser.parse_args()

    tables_dir, _ = out_dirs(load_config())

    if args.write_reference:
        REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
        for name in TABLES:
            shutil.copyfile(tables_dir / name, REFERENCE_DIR / name)
        print(f"saved {len(TABLES)} tables to {REFERENCE_DIR.relative_to(ROOT)}")
        return 0

    print("CHECK AGAINST THE PUBLISHED NUMBERS")
    problems = 0
    for name in TABLES:
        mine_path = tables_dir / name
        if not mine_path.exists():
            print(f"  MISSING  {name}: this step has not been run")
            problems += 1
            continue
        difference = compare(pd.read_csv(mine_path), pd.read_csv(REFERENCE_DIR / name))
        if difference is None:
            print(f"  PASS     {name}")
        else:
            print(f"  DIFFERS  {name}: {difference}")
            problems += 1

    if problems == 0:
        print(f"\nAll {len(TABLES)} tables agree with the published run.")
        return 0
    print(f"\n{problems} of {len(TABLES)} tables do not agree. A table that DIFFERS in the last "
          "digits only usually means different package versions: compare `pip freeze` with "
          "requirements.lock.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
