#!/usr/bin/env python
"""
05_preprocess_clinical.py  --  clinical block, encoding only.

    python prep/05_preprocess_clinical.py

Writes:
    data/processed/clinical_features.parquet   design matrix, NaNs PRESERVED
    results/tables/table3_cohort_characteristics.{csv,md}

Missing values are left as NaN on purpose. Iterative imputation is fitted
inside the inner training partition in 06_nested_cv.py; imputing here,
once, over 522 patients would let every test fold be filled using its own
values. That leak is small in effect and completely invisible to the
permutation control in 07, which is exactly what makes it worth avoiding.

What IS done here is fold-safe by construction:

  * Encoding      -- stage and smoking strings to ordinals via a fixed map.
  * Missingness indicators -- a per-patient fact, unchanged by which fold
    the patient lands in. They are carried alongside the imputed values so
    a model can learn that "stage not recorded" is itself informative,
    which in a deployment-missingness project is the whole point.
  * The structural-zero rule for pack-years (see below).

Structural zeros
----------------
All 75 lifelong non-smokers have pack_years = NA, and no non-smoker in the
cohort has a recorded value. That NA does not mean "unknown", it means
zero: the field is only ever filled in for smokers. Treating it as missing
would hand the imputer 75 fabricated smoking histories and would delete
the cleanest signal in the smoking variables. The rule is deterministic,
depends only on that patient's own record, and recovers 75 of 170 missing
pack-year values.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# Modelled columns, in the order they enter the design matrix.
FEATURES = ["age", "sex_male", "stage_ordinal", "smoking_ordinal",
            "ever_smoker", "pack_years"]

# Fields whose absence is plausibly informative rather than accidental.
# sex is never missing in TCGA-LUAD, so it gets no indicator.
INDICATED = ["age", "stage_ordinal", "smoking_ordinal", "pack_years"]


def main() -> int:
    cfg = yaml.safe_load(open("config/config.yaml"))
    interim = Path(cfg["paths"]["interim"])
    processed = Path(cfg["paths"]["processed"])
    tables = Path(cfg["paths"]["tables"])
    processed.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)

    clin_path = interim / "clinical_raw.parquet"
    endp_path = interim / "endpoints.parquet"
    for p in (clin_path, endp_path):
        if not p.exists():
            sys.exit(f"{p} not found.\n  run: Rscript prep/utils/export_clinical.R")

    print("=" * 62)
    print("CLINICAL BLOCK  --  encoding, no imputation")
    print("=" * 62)

    clin = pd.read_parquet(clin_path)
    endp = pd.read_parquet(endp_path)
    print(f"\npatients with covariates : {len(clin)}")
    print(f"patients with endpoints  : {len(endp)}")

    df = clin.set_index("patient_barcode").copy()

    # ---- structural zeros -------------------------------------------------
    never = (df["ever_smoker"] == 0)
    fill = never & df["pack_years"].isna()
    df.loc[fill, "pack_years"] = 0.0
    print(f"\npack-years: {int(fill.sum())} lifelong non-smokers set to a "
          f"structural 0")
    leftover = int(df["pack_years"].isna().sum())
    print(f"            {leftover} genuinely unknown remain "
          f"({leftover / len(df):.1%})")
    # If a non-smoker ever arrives with a recorded value, the field is not
    # what this rule assumes it is and the rule must be revisited.
    recorded = never & (df["pack_years"] > 0)
    if recorded.any():
        print(f"  !! {int(recorded.sum())} lifelong non-smokers have pack_years > 0 "
              f"-- the structural-zero rule no longer holds, re-check the source")

    # ---- missingness indicators -------------------------------------------
    print("\nmissingness (indicators added for the informative fields):")
    for col in FEATURES:
        n_miss = int(df[col].isna().sum())
        flag = "  <- indicator" if col in INDICATED else ""
        print(f"  {col:<16} {n_miss:>4} missing  ({n_miss / len(df):>5.1%}){flag}")
        if col in INDICATED:
            df[f"{col}_missing"] = df[col].isna().astype(np.int8)

    out_cols = FEATURES + [f"{c}_missing" for c in INDICATED]
    design = df[out_cols].astype(np.float64)

    # Every column must vary, or it contributes nothing and can destabilise
    # the imputer and the Cox fit.
    constant = [c for c in design.columns if design[c].nunique(dropna=True) <= 1]
    if constant:
        print(f"\n  !! constant columns (no information): {', '.join(constant)}")

    # ---- analysis cohort ---------------------------------------------------
    # The modelled set is patients with both covariates and a usable endpoint.
    # Complete covariates are NOT required: handling incomplete blocks is the
    # subject of the project, not an obstacle to it.
    analysis = design.index.intersection(endp.set_index("patient_barcode").index)
    print(f"\nanalysis cohort (covariates + endpoint): {len(analysis)}")
    n_complete = int(design.loc[analysis, FEATURES].notna().all(axis=1).sum())
    print(f"  complete clinical covariates          : {n_complete} "
          f"({n_complete / len(analysis):.1%})")
    print(f"  at least one covariate missing        : {len(analysis) - n_complete}")

    out = processed / "clinical_features.parquet"
    design.reset_index().to_parquet(out, index=False)

    write_table3(tables, df, endp, analysis)

    print(f"\n=== written ===")
    print(f"  {out}  ({len(design)} patients x {len(out_cols)} columns)")
    print("Next: python scripts/01_cv.py")
    return 0


def write_table3(tables, df, endp, analysis):
    """Baseline characteristics of the analysis cohort, split by PFI event.

    Descriptive only -- no test statistics. Comparing baseline covariates
    between event and non-event groups by p-value invites reading them as
    univariable prognostic findings, which is not what a table of cohort
    characteristics is for.
    """
    e = endp.set_index("patient_barcode").loc[analysis]
    d = df.loc[analysis]
    groups = {"Event (PFI = 1)": e["pfi_event"] == 1,
              "Censored (PFI = 0)": e["pfi_event"] == 0,
              "All": pd.Series(True, index=analysis)}

    rows = []

    def add_cont(label, col, fmt="{:.0f}"):
        r = {"Characteristic": label}
        for g, mask in groups.items():
            v = d.loc[mask, col].dropna()
            r[g] = (f"{fmt.format(v.median())} "
                    f"({fmt.format(v.quantile(.25))}–{fmt.format(v.quantile(.75))})")
        rows.append(r)

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

    rows.append({"Characteristic": "n", **{g: f"{int(m.sum())}" for g, m in groups.items()}})
    add_cont("Age at diagnosis, median (IQR)", "age")
    add_cat("Sex", "sex_male", {1.0: "Male", 0.0: "Female"})
    add_cat("Pathologic stage", "stage_ordinal",
            {1.0: "I", 2.0: "II", 3.0: "III", 4.0: "IV"})
    add_cat("Smoking status", "smoking_ordinal",
            {0.0: "Never", 1.0: "Former, >15 yr", 2.0: "Former, ≤15 yr", 3.0: "Current"})
    add_cont("Pack-years, median (IQR)", "pack_years")

    t3 = pd.DataFrame(rows)[["Characteristic", "All", "Event (PFI = 1)",
                             "Censored (PFI = 0)"]]
    t3.to_csv(tables / "table3_cohort_characteristics.csv", index=False)

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
