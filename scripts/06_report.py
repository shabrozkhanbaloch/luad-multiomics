#!/usr/bin/env python
"""Step 6. Write RESULTS.md from the tables. Nothing in it is typed by hand.

    python scripts/06_report.py

What it is for
    It collects the tables of the other steps into one readable document.
    Every number is read from results/tables/ at the moment this runs, so the
    report cannot disagree with the tables. A section whose step has not been
    run says so instead of being left out.

What it reads
    config/config.yaml                             the plan (endpoints, scenarios, folds)
    data/processed/, data/interim/                 only to count the patients and the events
    results/tables/arms_<endpoint>.csv             from step 2, for both endpoints
    results/tables/contrasts_<endpoint>.csv        from step 2, for both endpoints
    results/tables/external_arms.csv               from step 3
    results/tables/external_contrasts.csv          from step 3
    results/tables/biology_*.csv                   from steps 4 and 4b
    results/tables/checks.csv, width_sweep.csv     from step 5

What it writes
    RESULTS.md in the top folder of the project, replacing the old one. It
    has five sections: the two cross-validations, external validation,
    biology and checks.

How long it takes
    A few seconds. Nothing is fitted.

The script takes no options: it does the same thing whatever is typed after
its name.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

from _common import load_config, md_table, out_dirs

import pandas as pd

from luad import ROOT
from luad.data import load_cohort
from luad.models import MODEL_LABELS

# What a section says when the table it needs does not exist yet.
NOT_RUN = "_Not run yet._\n"
# The heading printed above each group of contrasts.
PART_TITLES = {"prediction": "Does a block add to clinical?",
               "integration": "Integration methods, nothing missing",
               "missing": "A block missing at test time",
               "control": "Pairing control (must be exactly zero)"}


def ci(v, lo, hi, signed=False) -> str:
    """A number with its interval as text, for example "0.598 (0.561, 0.634)".

    Arguments
        v        the estimate
        lo, hi   the lower and upper end of the interval
        signed   True to always print the sign, as wanted for a difference

    Returns
        The text "v (lo, hi)", each number with three decimal places.
    """
    number_format = "+.3f" if signed else ".3f"
    return f"{v:{number_format}} ({lo:{number_format}}, {hi:{number_format}})"


def read(tables: Path, name: str) -> pd.DataFrame | None:
    """Read one table, or return None if its step has not been run yet.

    Arguments
        tables   the folder of result tables
        name     the file name, for example "checks.csv"

    Returns
        The table, or None when the file does not exist. The caller then
        writes "Not run yet" instead of stopping with an error.
    """
    table_path = tables / name
    return pd.read_csv(table_path) if table_path.exists() else None


def arms_wide(arms: pd.DataFrame, keep: list[str] | None = None) -> pd.DataFrame:
    """One row per arm, one column per model: Uno's C with its interval.

    Arguments
        arms   the arms table of step 2 (one row per model, arm and metric)
        keep   the arms to show, in the order to show them; None shows all

    Returns
        A table with the column arm and one column per model, each cell
        holding text such as "0.598 (0.561, 0.634)".
    """
    uno_rows = arms[arms["metric"] == "uno_c"].copy()
    uno_rows["cell"] = [ci(row.c_index, row.ci_low, row.ci_high) for row in uno_rows.itertuples()]
    # Turn the long table (one row per model and arm) into a wide one, with
    # the models side by side under their full names.
    wide = uno_rows.pivot(index="arm", columns="model", values="cell")
    wide = wide.rename(columns=MODEL_LABELS)
    if keep:
        # An arm that was asked for but not run is skipped, not shown empty.
        wide = wide.reindex([arm for arm in keep if arm in wide.index])
    return wide.reset_index()


def contrast_table(contrasts: pd.DataFrame) -> pd.DataFrame:
    """The contrasts as they are shown in the report: one row per contrast and model.

    Arguments
        contrasts   rows of the contrasts table of step 2 (both metrics)

    Returns
        A table with the contrast in words, the model, the difference in
        Uno's C with its interval, the verdict, and the difference in
        Harrell's C beside it for comparison.
    """
    # Uno's C is the primary metric and gives the rows of the table.
    uno_rows = contrasts[contrasts["metric"] == "uno_c"].copy()
    # Harrell's C is looked up by model and arms, so that each value lands on
    # the row of the same contrast.
    harrell_difference = (contrasts[contrasts["metric"] == "harrell_c"]
                          .set_index(["model", "a", "b"])["difference"])
    return pd.DataFrame({
        "contrast (a minus b)": [f"{row.a} minus {row.b}" for row in uno_rows.itertuples()],
        "model": uno_rows["model"].map(MODEL_LABELS).to_numpy(),
        "difference in Uno's C (95% CI)": [ci(row.difference, row.ci_low, row.ci_high, True)
                                           for row in uno_rows.itertuples()],
        "verdict": uno_rows["verdict"].to_numpy(),
        "Harrell's C": [f"{harrell_difference.get((row.model, row.a, row.b), float('nan')):+.3f}"
                        for row in uno_rows.itertuples()],
    })


def section_cv(tables: Path, cfg: dict, endpoint: str) -> str:
    """The cross-validation section for one endpoint (Part 1, Part 2 and the tally).

    Arguments
        tables     the folder of result tables
        cfg        the plan; the scenarios give the rows of the Part 2 table
        endpoint   "pfi" or "os"

    Returns
        The markdown text of the section, or "Not run yet" if step 2 has not
        been run for this endpoint.
    """
    arms, contrasts = read(tables, f"arms_{endpoint}.csv"), read(tables, f"contrasts_{endpoint}.csv")
    if arms is None or contrasts is None:
        return NOT_RUN
    out = []

    # Part 1: the single blocks, each block added to clinical, and all four
    # together. First the concordances, then the contrasts against clinical.
    part1 = ["clinical", "rna", "mutation", "methylation", "clinical+rna",
             "clinical+mutation", "clinical+methylation", "early"]
    out += ["### Part 1. Prediction: clinical alone against clinical plus omics\n",
            "Uno's C with a 95% interval; `early` is all four blocks concatenated.\n",
            md_table(arms_wide(arms, part1)), ""]
    out += [md_table(contrast_table(contrasts[contrasts["part"] == "prediction"])), ""]

    # Part 2: one table per model. A row is a scenario (what the test patient
    # is missing), a column is an integration method.
    out += ["### Part 2. Integration: three methods, with and without a missing block\n",
            "Rows are what the test patient is missing. `early` imputes a missing block "
            "with training means; `late` drops that block's model; `mofa` projects the "
            "patient onto the factors from the blocks that are present.\n"]
    uno_rows = arms[arms["metric"] == "uno_c"]
    for model, label in MODEL_LABELS.items():
        model_rows = uno_rows[uno_rows["model"] == model].set_index("arm")
        rows = []
        for scenario in cfg["scenarios"]:
            row = {"missing at test": "nothing" if scenario == "full" else scenario.replace("_", " ")}
            for arm in ("early", "late", "mofa"):
                # In the tables an arm under a scenario is named arm@scenario;
                # with nothing missing it is just the arm.
                arm_key = arm if scenario == "full" else f"{arm}@{scenario}"
                row[arm] = (f"{model_rows.loc[arm_key, 'c_index']:.3f}"
                            if arm_key in model_rows.index else "n/a")
            # The clinical model is repeated on every row as the line to beat.
            row["clinical model"] = f"{model_rows.loc['clinical', 'c_index']:.3f}"
            rows.append(row)
        out += [f"**{label}**\n", md_table(pd.DataFrame(rows)), ""]
    for part in ("integration", "missing"):
        out += [f"**{PART_TITLES[part]}**\n",
                md_table(contrast_table(contrasts[contrasts["part"] == part])), ""]

    # The tally: how many contrasts came out each way, beside the number
    # that would look resolved by chance alone. The control is not a real
    # contrast, so it is counted apart and only checked for being zero.
    real = contrasts[(contrasts["part"] != "control") & (contrasts["metric"] == "uno_c")]
    counts = real["verdict"].value_counts()
    control = contrasts[contrasts["part"] == "control"]
    zero = bool(((control["difference"] == 0) & (control["ci_low"] == 0)
                 & (control["ci_high"] == 0)).all())
    out += [f"**Tally.** {len(real)} pre-declared contrasts (the list in `config/config.yaml`, "
            f"each under two models): {int(counts.get('a better', 0))} favour a, "
            f"{int(counts.get('b better', 0))} favour b, "
            f"{int(counts.get('not resolved', 0))} not resolved. At 95% intervals about "
            f"{0.05 * len(real):.0f} would resolve by chance alone. "
            f"Pairing control exactly zero: **{'yes' if zero else 'NO'}**.\n"]
    return "\n".join(out)


def section_external(tables: Path) -> str:
    """The external validation section: one table per model and a count of verdicts.

    Arguments
        tables   the folder of result tables

    Returns
        The markdown text of the section, or "Not run yet" if step 3 has not
        been run.
    """
    arms, contrasts = read(tables, "external_arms.csv"), read(tables, "external_contrasts.csv")
    if arms is None or contrasts is None:
        return NOT_RUN
    out = ["Models trained on TCGA and applied unchanged to GEO cohorts. GEO patients have "
           "expression and four clinical covariates only, so mutation and methylation are "
           "missing for real (`@geo`).\n"]
    uno_rows = arms[arms["metric"] == "uno_c"].copy()
    uno_rows["cell"] = [ci(row.c_index, row.ci_low, row.ci_high) for row in uno_rows.itertuples()]
    # The row label carries the cohort, the endpoint and its size, so each
    # line of the table can be read on its own.
    uno_rows["cohort"] = uno_rows["cohort"] + " " + uno_rows["endpoint"] \
        + " (n=" + uno_rows["n"].astype(str) + ", " \
        + uno_rows["n_events"].astype(str) + " events)"
    for model, label in MODEL_LABELS.items():
        wide = uno_rows[uno_rows["model"] == model].pivot(index="cohort", columns="arm", values="cell")
        # A fixed order of columns, from the simplest arm to the integration methods.
        wide = wide[[arm for arm in ["clinical", "clinical+rna", "early@geo", "late@geo", "mofa@geo"]
                     if arm in wide]]
        out += [f"**{label}**\n", md_table(wide.reset_index()), ""]
    # Each contrast was run in several cohorts, endpoints and models. Count
    # how often each verdict was reached, one row per contrast.
    uno_contrasts = contrasts[contrasts["metric"] == "uno_c"]
    verdict_counts = (uno_contrasts.groupby(["a", "b"], sort=False)["verdict"].value_counts()
                      .unstack(fill_value=0)
                      .reindex(columns=["a better", "not resolved", "b better"], fill_value=0)
                      .reset_index())
    verdict_counts.insert(0, "contrast (a minus b)",
                          verdict_counts.pop("a") + " minus " + verdict_counts.pop("b"))
    out += ["**Verdicts across cohorts, endpoints and models** (count of cohort-endpoint-model cells)\n",
            md_table(verdict_counts), ""]
    return "\n".join(out)


def section_biology(tables: Path) -> str:
    """The biology section: MOFA factors, gene lists, overlaps, the network and its modules.

    Arguments
        tables   the folder of result tables

    Returns
        The markdown text of the section, or "Not run yet" if step 4 has not
        been run. A single missing table inside the section is marked the
        same way.
    """
    factors = read(tables, "biology_factors.csv")
    if factors is None:
        return NOT_RUN
    out = ["### MOFA factors (whole cohort)\n",
           "`r2_*` is the share of a block's variance the factor explains; `cox_z` is its "
           "association with the endpoint (|z| > 2 is nominal); `r_*` are correlations.\n",
           md_table(factors, "+.2f"), ""]
    genes = read(tables, "biology_genes.csv")
    if genes is not None:
        # Each gene list is a true/false column of the genes table, so the
        # size of a list is the number of true values in its column.
        lists = [column for column in genes.columns if genes[column].dtype == bool]
        out += ["### Gene lists\n",
                md_table(pd.DataFrame({"list": lists,
                                       "genes": [int(genes[column].sum()) for column in lists]})), ""]
    # Four tables are shown the same way: a heading, one line of explanation
    # and the table itself.
    for name, title, note in [
        ("biology_overlap.csv", "Do the lists share genes?",
         "Hypergeometric test against all expressed genes."),
        ("biology_network_links.csv", "Are the lists linked in the protein network?",
         "Edges between two lists against the same count with labels shuffled."),
        ("biology_modules.csv", "Network modules", "Louvain modules of the STRING network."),
        ("biology_enrichment.csv", "Enrichment of the modules", "Top terms per module (STRING)."),
    ]:
        table = read(tables, name)
        out += [f"### {title}\n", note + "\n",
                md_table(table, ".3g") if table is not None else NOT_RUN, ""]
    centrality = read(tables, "biology_network_centrality.csv")
    out += ["### Best-connected genes (Cytoscape network analysis)\n",
            "Topology statistics computed by Cytoscape itself in step 4b; top 15 by degree.\n",
            md_table(centrality.head(15), ".3g") if centrality is not None else NOT_RUN, ""]
    out += ["Figures drawn by Cytoscape: `results/figures/network_cytoscape_lists.png` (colour: "
            "gene list) and `network_cytoscape_coxz.png` (colour: Cox z, blue protective to red "
            "risk). To explore: open `results/network/network.cys` in Cytoscape.\n"]
    return "\n".join(out)


def section_checks(tables: Path) -> str:
    """The checks section: the table of step 5, and the width sweep if it was run.

    Arguments
        tables   the folder of result tables

    Returns
        The markdown text of the section, or "Not run yet" if step 5 has not
        been run.
    """
    checks = read(tables, "checks.csv")
    if checks is None:
        return NOT_RUN
    out = [md_table(checks), ""]
    sweep = read(tables, "width_sweep.csv")
    if sweep is not None:
        out += ["**Width sweep** (fixed width instead of tuned; mean over folds)\n",
                md_table(sweep, "+.3f"), ""]
    return "\n".join(out)


def section_added(tables: Path, cfg: dict, endpoint: str) -> str:
    """The section for the two steps added after the first run (steps 10 and 11).

    Arguments
        tables    the folder of result tables
        cfg       the configuration, for the factor's name
        endpoint  the primary endpoint, lower case

    Returns
        The markdown text of the section; a part whose step has not been run
        says so.
    """
    out = ["_Steps 10 and 11 are not part of the plan fixed before the analysis. They were added "
           "afterwards, describe and explore, and add no contrast._\n",
           f"**Patients grouped by predicted risk, {endpoint.upper()}** (thirds of the out-of-fold "
           "risk; event-free share from Kaplan-Meier curves, with 95% intervals)\n"]
    groups = read(tables, f"survival_groups_{endpoint}.csv")
    if groups is None:
        out.append(NOT_RUN)
    else:
        shown = groups.assign(model=groups["model"].map(MODEL_LABELS))
        for column in [c for c in shown.columns if c.startswith("event_free_") and not c.endswith(("low", "high"))]:
            shown[column] = [ci(v, lo, hi) for v, lo, hi in
                             zip(shown[column], shown[f"{column}_low"], shown[f"{column}_high"])]
        shown = shown.drop(columns=[c for c in shown.columns if c.endswith(("_low", "_high"))])
        shown["logrank_p"] = shown["logrank_p"].map(lambda p: f"{p:.2g}")
        # Every cell is already text; `disable_numparse` stops the table writer
        # from reading "5.3e-06" back as a number and printing it as 0.000.
        out += [shown.drop(columns="logrank_chi2").to_markdown(index=False, disable_numparse=True), ""]

    factor = cfg["added_after_first_run"]["factor_biology"]["factor"]
    out.append(f"**What {factor} follows** (Pearson correlation with each programme, mutation and "
               "patient characteristic; FDR by Benjamini-Hochberg)\n")
    associations = read(tables, "biology_factor_associations.csv")
    if associations is None:
        out.append(NOT_RUN)
    else:
        shown = associations[["feature", "kind", "n", "r", "ci_low", "ci_high", "fdr"]].copy()
        for column in ("r", "ci_low", "ci_high"):
            shown[column] = shown[column].map(lambda v: f"{v:+.3f}")
        shown["fdr"] = shown["fdr"].map(lambda p: f"{p:.2g}")
        out += [shown.to_markdown(index=False, disable_numparse=True), ""]

    out.append(f"**Association of {factor} with {endpoint.upper()} after removing what a simpler "
               "variable explains**\n")
    outcome = read(tables, "biology_factor_outcome.csv")
    out += [NOT_RUN] if outcome is None else [md_table(outcome, "+.3f"), ""]
    return "\n".join(out)


def main() -> int:
    """Build the report from the tables and write RESULTS.md.

    Returns
        0, the exit code for success.
    """
    # ---- Read the plan ----
    cfg = load_config()
    tables, _ = out_dirs(cfg)
    primary = cfg["endpoint"]["primary"].lower()
    sensitivity = cfg["endpoint"]["sensitivity"].lower()

    # ---- Load the data ----
    # The cohort is loaded only for two numbers in the opening paragraph: how
    # many patients and how many events.
    cohort = load_cohort(cfg, primary, verbose=False)

    # ---- Compute: put the document together, one piece of text per item ----
    # The date is on a line of its own (the one starting with "_Generated"),
    # so two reports can be compared without it.
    doc = [
        "# Results\n",
        f"_Generated by `scripts/06_report.py` on {date.today().isoformat()} from "
        "`results/tables/`. Do not edit by hand; rerun the script._\n",
        f"**Cohort.** {cohort.n} TCGA-LUAD patients with an endpoint and all four blocks "
        f"(clinical, expression, mutation, methylation). Primary endpoint {primary.upper()}, "
        f"censored at {cfg['endpoint']['admin_censor_months']} months: "
        f"{int(cohort.y['event'].sum())} events. "
        f"{cfg['cv']['outer_folds']}-fold cross-validation repeated {cfg['cv']['outer_repeats']} "
        "times; intervals from a paired patient-level bootstrap "
        f"({cfg['bootstrap']['n_resamples']} resamples).\n",
        f"## 1. Cross-validation, {primary.upper()} (primary)\n", section_cv(tables, cfg, primary),
        f"## 2. Cross-validation, {sensitivity.upper()} (sensitivity)\n",
        section_cv(tables, cfg, sensitivity),
        "## 3. External validation\n", section_external(tables),
        "## 4. Biology\n", section_biology(tables),
        "## 5. Checks\n", section_checks(tables),
        "## 6. Added after the first run\n", section_added(tables, cfg, primary),
    ]

    # ---- Write the report ----
    path = ROOT / "RESULTS.md"
    path.write_text("\n".join(doc))

    # ---- Print the summary ----
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
