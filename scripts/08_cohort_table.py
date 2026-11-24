#!/usr/bin/env python
"""Step 8. Baseline characteristics of the patients that were analysed.

    python scripts/08_cohort_table.py

What it is for
    The analysis cohort is the patients with an endpoint and all four blocks
    (440 for the primary endpoint). `prep/05_preprocess_clinical.py` writes a
    similar table, but for everyone with clinical covariates and an endpoint
    (509), which is the cohort before the four-block rule is applied. This
    script describes the cohort the models actually saw, with events counted
    under the same 60-month censoring the analysis uses.

    Descriptive only: no test statistics are reported.

What it reads
    config/config.yaml                          the endpoint and the censoring time
    data/processed/clinical_features.parquet    age, sex, stage and smoking of each patient
    data/processed/, data/interim/              to find the patients of the analysis cohort

What it writes
    results/tables/cohort_characteristics.csv   the table: all patients, progressed, censored
    results/tables/cohort_characteristics.md    the same table as markdown, with a short heading
    results/tables/cohort_flow.csv              how 509 patients become 440, one row per rule

How long it takes
    A few seconds. Nothing is fitted.

The script takes no options: it does the same thing whatever is typed after
its name.
"""
from __future__ import annotations

from pathlib import Path

from _common import load_config, md_table, out_dirs

import pandas as pd

from luad.data import cohort_flow, load_cohort

# The clinical file stores categories as numbers. These give each number its
# label, in the order the rows are printed.
SEX = {1.0: "Male", 0.0: "Female"}
STAGE = {1.0: "I", 2.0: "II", 3.0: "III", 4.0: "IV"}
SMOKING = {0.0: "Never", 1.0: "Former, >15 yr", 2.0: "Former, ≤15 yr", 3.0: "Current"}


def build_table(clin: pd.DataFrame, event: pd.Series) -> pd.DataFrame:
    """One row per characteristic; columns All, Progressed, Censored.

    Arguments
        clin    the clinical covariates, one row per patient of the cohort
        event   True for a patient who had the event (progressed), False for
                a censored patient; same row order as `clin`

    Returns
        The table as text cells: a median with its quartiles for a number,
        a count with its percentage for a category.
    """
    # Each column of the table is a group of patients, held as a true/false
    # mask over the rows of `clin`.
    groups = {"All": pd.Series(True, index=clin.index),
              "Progressed": event,
              "Censored": ~event}
    # The first row is the size of each group.
    rows = [{"Characteristic": "n",
             **{group: str(int(mask.sum())) for group, mask in groups.items()}}]

    def continuous(label: str, col: str) -> None:
        """Add one row for a numeric characteristic: median (lower to upper quartile).

        Arguments
            label   the text shown in the first column
            col     the column of `clin` to summarise

        Returns
            Nothing. The row is added to `rows`.
        """
        row = {"Characteristic": label}
        for group, mask in groups.items():
            # Patients with no value recorded are left out of the median.
            values = clin.loc[mask, col].dropna()
            row[group] = (f"{values.median():.0f} "
                          f"({values.quantile(.25):.0f}–{values.quantile(.75):.0f})")
        rows.append(row)

    def categorical(label: str, col: str, levels: dict[float, str]) -> None:
        """Add the rows for a category: a heading, one row per level, and "not recorded".

        Arguments
            label    the heading shown in the first column
            col      the column of `clin` to count
            levels   the stored number of each level and its label

        Returns
            Nothing. The rows are added to `rows`.
        """
        rows.append({"Characteristic": label, **{group: "" for group in groups}})
        for value, name in levels.items():
            row = {"Characteristic": f"    {name}"}
            for group, mask in groups.items():
                # The percentage is out of everyone in the group, including
                # patients with no value recorded.
                group_values = clin.loc[mask, col]
                count = int((group_values == value).sum())
                row[group] = f"{count} ({count / len(group_values):.1%})"
            rows.append(row)
        # Missing values get a row of their own, so the percentages of a
        # category add up to 100.
        row = {"Characteristic": "    not recorded"}
        for group, mask in groups.items():
            group_values = clin.loc[mask, col]
            count = int(group_values.isna().sum())
            row[group] = f"{count} ({count / len(group_values):.1%})"
        rows.append(row)

    # The rows, in the order they appear in the table.
    continuous("Age at diagnosis, median (IQR)", "age")
    categorical("Sex", "sex_male", SEX)
    categorical("Pathologic stage", "stage_ordinal", STAGE)
    categorical("Smoking status", "smoking_ordinal", SMOKING)
    continuous("Pack-years, median (IQR)", "pack_years")
    return pd.DataFrame(rows)[["Characteristic", "All", "Progressed", "Censored"]]


def main() -> int:
    """Write the baseline table (CSV and markdown) and the cohort flow table.

    Returns
        0, the exit code for success.
    """
    # ---- Read the plan ----
    cfg = load_config()
    tables, _ = out_dirs(cfg)

    # ---- Load the data ----
    cohort = load_cohort(cfg)

    # The cohort holds the clinical data as a bare array of numbers. For a
    # table with named columns the clinical file is read again, and its rows
    # are put in the order of the analysis cohort so that they line up with
    # the events.
    clin = (pd.read_parquet(Path(cfg["paths"]["processed"]) / "clinical_features.parquet")
            .set_index("patient_barcode").loc[cohort.patients])
    event = pd.Series(cohort.y["event"], index=clin.index)

    # ---- Compute and write the baseline table ----
    table = build_table(clin, event)
    table.to_csv(tables / "cohort_characteristics.csv", index=False)

    endpoint = cfg["endpoint"]["primary"]
    months = cfg["endpoint"]["admin_censor_months"]
    with open(tables / "cohort_characteristics.md", "w") as fh:
        fh.write("# Baseline characteristics of the analysis cohort\n\n")
        fh.write(f"TCGA-LUAD, {cohort.n} patients with an endpoint and all four blocks.\n")
        fh.write(f"{int(event.sum())} {endpoint} events ({event.mean():.1%}) "
                 f"within {months} months.\n\n")
        fh.write("Descriptive only; no between-group tests are reported.\n\n")
        fh.write(md_table(table))
        fh.write("\n")
    print(f"  wrote {tables / 'cohort_characteristics.md'}")

    # ---- Compute and write the cohort flow ----
    # The flow table counts the patients left after each inclusion rule. Its
    # last row must be the cohort that was analysed; if not, the two were
    # built from different rules and the run stops.
    flow = cohort_flow(cfg)
    assert int(flow["patients"].iloc[-1]) == cohort.n, "flow table and cohort disagree"
    flow.to_csv(tables / "cohort_flow.csv", index=False)

    # ---- Print the summary ----
    print(flow.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
