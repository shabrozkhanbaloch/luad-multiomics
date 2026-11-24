#!/usr/bin/env python
"""Step 8. Baseline characteristics of the patients that were analysed.

    python scripts/08_cohort_table.py

The analysis cohort is the patients with an endpoint and all four blocks
(440 for the primary endpoint). `prep/05_preprocess_clinical.py` writes a
similar table, but for everyone with clinical covariates and an endpoint
(509), which is the cohort before the four-block rule is applied. This
script describes the cohort the models actually saw, with events counted
under the same 60-month censoring the analysis uses.

Descriptive only: no test statistics are reported.

Writes results/tables/cohort_characteristics.csv and .md
"""
from __future__ import annotations

from pathlib import Path

from _common import load_config, md_table, out_dirs

import pandas as pd

from luad.data import cohort_flow, load_cohort

SEX = {1.0: "Male", 0.0: "Female"}
STAGE = {1.0: "I", 2.0: "II", 3.0: "III", 4.0: "IV"}
SMOKING = {0.0: "Never", 1.0: "Former, >15 yr", 2.0: "Former, ≤15 yr", 3.0: "Current"}


def build_table(clin: pd.DataFrame, event: pd.Series) -> pd.DataFrame:
    """One row per characteristic; columns All, Progressed, Censored."""
    groups = {"All": pd.Series(True, index=clin.index),
              "Progressed": event,
              "Censored": ~event}
    rows = [{"Characteristic": "n", **{g: str(int(m.sum())) for g, m in groups.items()}}]

    def continuous(label: str, col: str) -> None:
        row = {"Characteristic": label}
        for g, mask in groups.items():
            v = clin.loc[mask, col].dropna()
            row[g] = f"{v.median():.0f} ({v.quantile(.25):.0f}–{v.quantile(.75):.0f})"
        rows.append(row)

    def categorical(label: str, col: str, levels: dict[float, str]) -> None:
        rows.append({"Characteristic": label, **{g: "" for g in groups}})
        for value, name in levels.items():
            row = {"Characteristic": f"    {name}"}
            for g, mask in groups.items():
                sub = clin.loc[mask, col]
                n = int((sub == value).sum())
                row[g] = f"{n} ({n / len(sub):.1%})"
            rows.append(row)
        row = {"Characteristic": "    not recorded"}
        for g, mask in groups.items():
            sub = clin.loc[mask, col]
            n = int(sub.isna().sum())
            row[g] = f"{n} ({n / len(sub):.1%})"
        rows.append(row)

    continuous("Age at diagnosis, median (IQR)", "age")
    categorical("Sex", "sex_male", SEX)
    categorical("Pathologic stage", "stage_ordinal", STAGE)
    categorical("Smoking status", "smoking_ordinal", SMOKING)
    continuous("Pack-years, median (IQR)", "pack_years")
    return pd.DataFrame(rows)[["Characteristic", "All", "Progressed", "Censored"]]


def main() -> int:
    cfg = load_config()
    tables, _ = out_dirs(cfg)
    cohort = load_cohort(cfg)

    clin = (pd.read_parquet(Path(cfg["paths"]["processed"]) / "clinical_features.parquet")
            .set_index("patient_barcode").loc[cohort.patients])
    event = pd.Series(cohort.y["event"], index=clin.index)

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

    flow = cohort_flow(cfg)
    assert int(flow["patients"].iloc[-1]) == cohort.n, "flow table and cohort disagree"
    flow.to_csv(tables / "cohort_flow.csv", index=False)
    print(flow.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
