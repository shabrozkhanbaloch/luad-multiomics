#!/usr/bin/env python
"""
05_preprocess_clinical.py -- the clinical block.

What it is for
    Turns the per-patient clinical record into the table of clinical
    covariates the models read: six covariates and four 0/1 columns that
    say "this value was missing". Together they are the ten clinical
    covariates of the README. It also writes a table of baseline
    characteristics.

Where it sits
    Step 7 of 9 of `make prep`. It needs prep/utils/export_clinical.R
    (step 6), which writes both files it reads.

Reads
    data/interim/clinical_raw.parquet   covariates, one row per patient
    data/interim/endpoints.parquet      the outcome (used for the table and
                                        for counting, not for the covariates)
    config/config.yaml                  (folders)

Writes
    data/processed/clinical_features.parquet   one row per patient, missing
                                               values kept as missing (NaN)
    results/tables/table3_cohort_characteristics.csv and .md

How to run (from the project folder, with the project's Python)
    .venv/bin/python prep/05_preprocess_clinical.py

The README states that this step was rerun in this repository and
reproduced the files in data/processed exactly.

Why missing values are not filled in here
-----------------------------------------
Missing values are left as NaN on purpose. They are filled in (imputed)
later, by a method that is fitted on the training patients of each
cross-validation fold (src/luad/transforms.py). Imputing here, once, over
all 522 patients would let the patients held out for testing be filled in
using their own values. That leak is small in effect and completely
invisible to the shuffled-outcome check (`make permute`), which is exactly
what makes it worth avoiding.

What IS done here depends only on each patient's own record, so it is the
same whichever fold the patient later lands in:

  * Encoding. Stage and smoking were turned from text into ordered
    numbers by fixed tables in prep/utils/export_clinical.R; they arrive
    here as numbers.
  * Missingness indicators. A column that is 1 where a value was missing
    and 0 where it was recorded. It is kept beside the imputed value so
    that a model can learn that "stage not recorded" is itself
    informative.
  * The structural-zero rule for pack-years (see below).

Structural zeros
----------------
Pack-years measure lifetime smoking: packs per day multiplied by years
smoked. All 75 lifelong non-smokers have pack_years = NA, and no
non-smoker in the cohort has a recorded value. That NA does not mean
"unknown", it means zero: the field is only ever filled in for smokers.
Treating it as missing would hand the imputer 75 fabricated smoking
histories and would delete the cleanest signal in the smoking variables.
The rule is deterministic, depends only on that patient's own record, and
recovers 75 of 170 missing pack-year values.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# The six covariates, in the order of the columns of the output table:
# age in years, sex (1 = male), stage (1 to 4), smoking (0 = never to
# 3 = current), ever smoked (0 or 1), and pack-years.
FEATURES = ["age", "sex_male", "stage_ordinal", "smoking_ordinal",
            "ever_smoker", "pack_years"]

# The covariates that get a "was missing" column: those where a missing
# value may itself say something about the patient.
# Sex is never missing in TCGA-LUAD, so it gets no indicator. ever_smoker
# has no indicator of its own either.
INDICATED = ["age", "stage_ordinal", "smoking_ordinal", "pack_years"]


def main() -> int:
    """Build the covariate table and the baseline table; return 0 on success."""
    # Folders from the configuration; create the two output folders.
    cfg = yaml.safe_load(open("config/config.yaml"))
    interim = Path(cfg["paths"]["interim"])
    processed = Path(cfg["paths"]["processed"])
    tables = Path(cfg["paths"]["tables"])
    processed.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)

    # Both inputs come from prep/utils/export_clinical.R.
    clin_path = interim / "clinical_raw.parquet"
    endp_path = interim / "endpoints.parquet"
    for p in (clin_path, endp_path):
        if not p.exists():
            sys.exit(f"{p} not found.\n  run: Rscript prep/utils/export_clinical.R")

    print("=" * 62)
    print("CLINICAL BLOCK  --  encoding, no imputation")
    print("=" * 62)

    # Covariates (one row per patient) and endpoints (one row per patient
    # with a usable progression-free interval).
    clin = pd.read_parquet(clin_path)
    endp = pd.read_parquet(endp_path)
    print(f"\npatients with covariates : {len(clin)}")
    print(f"patients with endpoints  : {len(endp)}")

    # Work on a copy, with the patient barcode as the row label.
    df = clin.set_index("patient_barcode").copy()

    # ---- structural zeros -------------------------------------------------
    # Lifelong non-smokers with no pack-years recorded get 0 pack-years (the
    # reason is in the header). What is still missing afterwards is
    # really unknown, and its count is printed.
    never = (df["ever_smoker"] == 0)
    fill = never & df["pack_years"].isna()
    df.loc[fill, "pack_years"] = 0.0
    print(f"\npack-years: {int(fill.sum())} lifelong non-smokers set to a "
          f"structural 0")
    leftover = int(df["pack_years"].isna().sum())
    print(f"            {leftover} genuinely unknown remain "
          f"({leftover / len(df):.1%})")
    # The rule assumes that pack-years are never recorded for non-smokers.
    # If a non-smoker has a value above zero, that assumption is wrong for
    # this data and the rule must be looked at again; a warning is printed.
    recorded = never & (df["pack_years"] > 0)
    if recorded.any():
        print(f"  !! {int(recorded.sum())} lifelong non-smokers have pack_years > 0 "
              f"-- the structural-zero rule no longer holds, re-check the source")

    # ---- missingness indicators -------------------------------------------
    # Print how many values each covariate is missing, and add the 0/1
    # column <name>_missing for the covariates in INDICATED. This is done
    # after the structural zeros, so a non-smoker's pack-years do not count
    # as missing.
    print("\nmissingness (indicators added for the informative fields):")
    for col in FEATURES:
        n_miss = int(df[col].isna().sum())
        flag = "  <- indicator" if col in INDICATED else ""
        print(f"  {col:<16} {n_miss:>4} missing  ({n_miss / len(df):>5.1%}){flag}")
        if col in INDICATED:
            df[f"{col}_missing"] = df[col].isna().astype(np.int8)

    # The output table ("design matrix": one row per patient, one column
    # per covariate): the six covariates, then the four indicators.
    out_cols = FEATURES + [f"{c}_missing" for c in INDICATED]
    design = df[out_cols].astype(np.float64)

    # Every column must vary between patients. A column with one value
    # carries no information and can upset the imputation and the Cox
    # model, so such columns are named in a warning.
    constant = [c for c in design.columns if design[c].nunique(dropna=True) <= 1]
    if constant:
        print(f"\n  !! constant columns (no information): {', '.join(constant)}")

    # ---- patients with covariates and an endpoint -------------------------
    # Counted here for the log and the baseline table only. The variable is called
    # `analysis`, but this is the source cohort: the cohort the analysis
    # uses is formed later, in src/luad/data.py, and also requires the
    # expression, mutation and methylation blocks (440 patients).
    # Complete covariates are NOT required. A patient with a missing value
    # stays in, and the value is imputed inside each fold.
    analysis = design.index.intersection(endp.set_index("patient_barcode").index)
    print(f"\nanalysis cohort (covariates + endpoint): {len(analysis)}")
    n_complete = int(design.loc[analysis, FEATURES].notna().all(axis=1).sum())
    print(f"  complete clinical covariates          : {n_complete} "
          f"({n_complete / len(analysis):.1%})")
    print(f"  at least one covariate missing        : {len(analysis) - n_complete}")

    # Write the covariates of ALL patients in the covariate table, not only
    # of those with an endpoint. The analysis makes its own selection.
    out = processed / "clinical_features.parquet"
    design.reset_index().to_parquet(out, index=False)

    write_table3(tables, df, endp, analysis)

    print(f"\n=== written ===")
    print(f"  {out}  ({len(design)} patients x {len(out_cols)} columns)")
    print("Next: python scripts/01_cv.py")
    return 0


def write_table3(tables, df, endp, analysis):
    """Baseline characteristics of the source cohort, split by PFI event.

    The source cohort is every patient with clinical covariates and a usable
    progression-free interval (PFI), before the analysis cohort is formed.
    The two groups are patients in whom the event was seen (PFI = 1) and
    patients who were censored (PFI = 0): their follow-up ended without
    the event.

    Descriptive only: no test statistics. Comparing baseline covariates
    between the two groups by p-value invites reading them as findings
    about prognosis, which is not what a table of cohort characteristics
    is for.
    """
    # `e` holds the endpoints and `d` the covariates of the same patients.
    # `groups` maps each column heading of the table to the patients in it.
    e = endp.set_index("patient_barcode").loc[analysis]
    d = df.loc[analysis]
    groups = {"Event (PFI = 1)": e["pfi_event"] == 1,
              "Censored (PFI = 0)": e["pfi_event"] == 0,
              "All": pd.Series(True, index=analysis)}

    rows = []

    # A continuous covariate becomes one row: the median, with the
    # interquartile range (25th to 75th percentile) in brackets.
    def add_cont(label, col, fmt="{:.0f}"):
        r = {"Characteristic": label}
        for g, mask in groups.items():
            v = d.loc[mask, col].dropna()
            r[g] = (f"{fmt.format(v.median())} "
                    f"({fmt.format(v.quantile(.25))}–{fmt.format(v.quantile(.75))})")
        rows.append(r)

    # A categorical covariate becomes a heading row, one row per level
    # with the count and its share of the group, and a "not recorded" row.
    def add_cat(label, col, levels):
        rows.append({"Characteristic": label,
                     **{g: "" for g in groups}})
        for lv, name in levels.items():
            r = {"Characteristic": f"    {name}"}
            for g, mask in groups.items():
                sub = d.loc[mask, col]
                n = int((sub == lv).sum())
                r[g] = f"{n} ({n / len(sub):.1%})"
            rows.append(r)
        r = {"Characteristic": "    not recorded"}
        for g, mask in groups.items():
            n = int(d.loc[mask, col].isna().sum())
            r[g] = f"{n} ({n / int(mask.sum()):.1%})"
        rows.append(r)

    # The rows of the table, from top to bottom.
    rows.append({"Characteristic": "n", **{g: f"{int(m.sum())}" for g, m in groups.items()}})
    add_cont("Age at diagnosis, median (IQR)", "age")
    add_cat("Sex", "sex_male", {1.0: "Male", 0.0: "Female"})
    add_cat("Pathologic stage", "stage_ordinal",
            {1.0: "I", 2.0: "II", 3.0: "III", 4.0: "IV"})
    add_cat("Smoking status", "smoking_ordinal",
            {0.0: "Never", 1.0: "Former, >15 yr", 2.0: "Former, ≤15 yr", 3.0: "Current"})
    add_cont("Pack-years, median (IQR)", "pack_years")

    # Put the columns in reading order and write the CSV.
    t3 = pd.DataFrame(rows)[["Characteristic", "All", "Event (PFI = 1)",
                             "Censored (PFI = 0)"]]
    t3.to_csv(tables / "table3_cohort_characteristics.csv", index=False)

    # The markdown version adds a short description above the table.
    # Follow-up is given in years (days / 365.25) among censored patients.
    pfi_yrs = (e["pfi_time"] / 365.25)
    with open(tables / "table3_cohort_characteristics.md", "w") as fh:
        fh.write("# Table 3. Baseline characteristics of the source cohort\n\n")
        fh.write(f"TCGA-LUAD, {len(analysis)} patients with clinical covariates "
                 f"and a usable progression-free interval.\n")
        fh.write(f"{int((e['pfi_event'] == 1).sum())} PFI events "
                 f"({(e['pfi_event'] == 1).mean():.1%}); median follow-up "
                 f"{pfi_yrs[e['pfi_event'] == 0].median():.1f} years among the "
                 f"censored.\n\n")
        fh.write("> This is the source cohort, before the analysis cohort is formed "
                 "and before follow-up is censored at 60 months. The analysis uses the "
                 "440 of these patients who have all four blocks (176 events within 60 "
                 "months); see cohort_flow.csv and cohort_characteristics.md.\n\n")
        fh.write("Descriptive only; no between-group tests are reported, because "
                 "a table of\ncohort characteristics is not a univariable "
                 "prognostic analysis.\n\n")
        fh.write(t3.to_markdown(index=False))
        fh.write("\n")
    print(f"\n  wrote {tables / 'table3_cohort_characteristics.md'}")


if __name__ == "__main__":
    sys.exit(main())
