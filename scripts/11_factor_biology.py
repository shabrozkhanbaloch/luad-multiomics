#!/usr/bin/env python
"""Step 11. What is the MOFA factor that is linked to outcome?

    python scripts/11_factor_biology.py

ADDED AFTER THE FIRST RUN. This step is not part of the plan that was fixed in
config/config.yaml before the analysis. It was added on 24 November 2026, after
the results were known. It is exploratory, like the rest of the biology part:
it describes one factor, tests nothing that was declared in advance, and no
claim about prediction rests on it.

What it is for
    Step 4 found that one of the ten MOFA factors is associated with
    progression (factor 2, |Cox z| above 2), and that it follows tumour
    mutation burden. A factor is only a column of numbers, one per patient.
    This step asks what those numbers mean biologically, in three ways:

    1. Which biological programmes does the factor follow? Each programme
       (proliferation, immune cells, fibroblasts, ...) is a short list of
       marker genes in config.yaml; a patient's score for a programme is the
       average of the standardised expression of those genes.
    2. Which driver mutations and patient characteristics does it follow?
    3. Is its link to outcome only a reflection of something simpler? The
       factor's association with the endpoint is measured again after the
       part of it that a simpler variable explains has been removed.

What it reads
    config/config.yaml                        the factor, the marker sets, the driver genes
    results/tables/biology_factor_scores.csv  each patient's value on each factor (step 4)
    data/processed/, data/interim/            expression, mutations, clinical covariates, outcome

What it writes
    results/tables/biology_factor_associations.csv  one row per programme, mutation or
                                                    characteristic: its correlation with the factor
    results/tables/biology_factor_outcome.csv       the factor's association with the endpoint,
                                                    alone and after removing each simpler variable

How long it takes
    A few seconds. Nothing is fitted except straight-line regressions.

Words used here
    correlation (r)  from -1 to +1: how closely two measurements move together
    Cox z            how strongly a variable is associated with the endpoint, in
                     standard errors; beyond about 2 either way is more than chance
                     usually gives. Negative: higher values go with a better outcome.
    FDR              a p-value corrected for the number of tests in the table
                     (Benjamini-Hochberg)
"""
from __future__ import annotations

from _common import load_config, out_dirs

import numpy as np
import pandas as pd
from scipy import stats

from luad import biology as B
from luad.data import load_cohort

CLINICAL_LABELS = {"stage_ordinal": "Stage", "age": "Age", "sex_male": "Male sex",
                   "ever_smoker": "Ever smoker", "pack_years": "Pack-years"}


def standardise(column: np.ndarray) -> np.ndarray:
    """Subtract the mean and divide by the standard deviation."""
    spread = column.std()
    return (column - column.mean()) / (spread if spread > 1e-12 else 1.0)


def programme_scores(cohort, signatures: dict[str, list[str]]) -> tuple[pd.DataFrame, dict[str, int]]:
    """One score per patient and programme.

    A gene's expression is first standardised across patients, so that a
    highly expressed gene does not dominate its set. The score is the mean
    over the set's genes that are in the expression data.

    Returns
        scores  patients (rows) by programmes (columns)
        found   how many of each set's genes were in the data
    """
    # Where each gene symbol sits in the expression table. If a symbol occurs
    # twice, the first occurrence is used, as in step 4.
    column_of = {}
    for position, gene in enumerate(cohort.rna_genes):
        column_of.setdefault(gene, position)

    scores, found = {}, {}
    for name, genes in signatures.items():
        present = [gene for gene in genes if gene in column_of]
        found[name] = len(present)
        standardised = np.column_stack([standardise(cohort.rna[:, column_of[gene]].astype(float))
                                        for gene in present])
        scores[name] = standardised.mean(axis=1)
    return pd.DataFrame(scores), found


def correlation(factor: np.ndarray, other: np.ndarray) -> dict:
    """Pearson correlation of the factor with one variable, on the patients who have both.

    Returns the number of patients, r, its 95% interval and its p-value. The
    interval uses Fisher's transformation, the usual formula for a correlation.
    """
    both = ~np.isnan(factor) & ~np.isnan(other)
    r, p_value = stats.pearsonr(factor[both], other[both])
    n = int(both.sum())
    half_width = 1.96 / np.sqrt(n - 3)
    return {"n": n, "r": r, "ci_low": np.tanh(np.arctanh(r) - half_width),
            "ci_high": np.tanh(np.arctanh(r) + half_width), "p_value": p_value}


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """P-values corrected for the number of tests, so that about 5% of the
    rows called significant at 0.05 are expected to be false alarms."""
    n = len(p_values)
    order = np.argsort(p_values)
    scaled = p_values[order] * n / np.arange(1, n + 1)
    # Each corrected value may not be smaller than the one before it in rank.
    corrected = np.minimum.accumulate(scaled[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(corrected, 1.0)
    return out


def remove_explained_part(factor: np.ndarray, explain_with: np.ndarray) -> np.ndarray:
    """What is left of the factor after a straight-line fit on other variables.

    Arguments
        factor        one value per patient
        explain_with  patients (rows) by variables (columns), no missing values

    Returns the residual: the part of the factor those variables cannot explain.
    """
    design = np.column_stack([np.ones(len(factor)), explain_with])
    coefficients, *_ = np.linalg.lstsq(design, factor, rcond=None)
    return factor - design @ coefficients


def main() -> int:
    """Describe the factor and write the two tables."""
    # ---- Read the plan ----
    cfg = load_config()
    settings = cfg["added_after_first_run"]["factor_biology"]
    factor_name = settings["factor"]
    endpoint = cfg["endpoint"]["primary"].lower()
    tables, _ = out_dirs(cfg)

    # ---- Load the data ----
    print(f"WHAT IS {factor_name.upper()}?")
    cohort = load_cohort(cfg, endpoint)
    factor_scores = pd.read_csv(tables / "biology_factor_scores.csv").set_index("patient")
    # The two tables must describe the same patients in the same order.
    assert list(factor_scores.index) == list(cohort.patients), "factor scores and cohort disagree"
    factor = factor_scores[factor_name].to_numpy()

    clinical = pd.DataFrame(cohort.clinical, columns=cohort.clinical_names)
    burden = cohort.mut_feat[:, cohort.mut_feat_names.index("tmb_log10")].astype(float)
    programmes, genes_found = programme_scores(cohort, settings["signatures"])

    # ---- Compute 1 and 2: what does the factor follow? ----
    rows = []
    for name in programmes:
        rows.append({"feature": name, "kind": "programme", **correlation(factor, programmes[name].to_numpy()),
                     "genes_in_set": len(settings["signatures"][name]), "genes_found": genes_found[name]})
    for gene in settings["driver_genes"]:
        mutated = cohort.mut_gene[:, list(cohort.mut_gene_names).index(gene)].astype(float)
        rows.append({"feature": f"{gene} mutated", "kind": "mutation", **correlation(factor, mutated),
                     "share_with_feature": mutated.mean()})
    rows.append({"feature": "Mutation burden", "kind": "mutation", **correlation(factor, burden)})
    for column, label in CLINICAL_LABELS.items():
        rows.append({"feature": label, "kind": "clinical",
                     **correlation(factor, clinical[column].to_numpy(float))})
    associations = pd.DataFrame(rows)
    # Counts are whole numbers; rows they do not apply to are left empty.
    for column in ("genes_in_set", "genes_found"):
        associations[column] = associations[column].astype("Int64")
    associations["fdr"] = benjamini_hochberg(associations["p_value"].to_numpy())
    # Strongest first, whatever the sign.
    associations = (associations.reindex(associations["r"].abs().sort_values(ascending=False).index)
                    .reset_index(drop=True))

    # ---- Compute 3: is the link to outcome a reflection of something simpler? ----
    # Missing clinical values are replaced by the column's median so that every
    # patient can be used; this is a description, not a model that is scored.
    clinical_filled = clinical[list(CLINICAL_LABELS)].apply(lambda column: column.fillna(column.median()))
    proliferation = programmes["Cell cycle and proliferation"].to_numpy()
    simpler = {
        "nothing": None,
        "Cell cycle and proliferation": proliferation[:, None],
        "Mutation burden": burden[:, None],
        "Stage": clinical_filled[["stage_ordinal"]].to_numpy(float),
        "All clinical covariates": clinical_filled.to_numpy(float),
        "Proliferation, mutation burden and clinical covariates":
            np.column_stack([proliferation, burden, clinical_filled.to_numpy(float)]),
    }
    outcome_rows = []
    for label, variables in simpler.items():
        remaining = factor if variables is None else remove_explained_part(factor, variables)
        cox_z = float(B.cox_score_z(remaining[:, None], cohort.y)[0])
        # How much of the factor's variation the simpler variables account for.
        explained = 0.0 if variables is None else 1.0 - remaining.var() / factor.var()
        outcome_rows.append({"after_removing": label, "share_of_factor_explained": explained, "cox_z": cox_z})
    outcome = pd.DataFrame(outcome_rows)

    # ---- Write the tables ----
    associations.to_csv(tables / "biology_factor_associations.csv", index=False)
    outcome.to_csv(tables / "biology_factor_outcome.csv", index=False)

    # ---- Print the summary ----
    print(associations[["feature", "kind", "n", "r", "ci_low", "ci_high", "fdr"]]
          .to_string(index=False, float_format=lambda v: f"{v:+.3f}"))
    print()
    print(outcome.to_string(index=False, float_format=lambda v: f"{v:+.3f}"))
    print(f"\n  wrote {tables}/biology_factor_associations.csv and biology_factor_outcome.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
