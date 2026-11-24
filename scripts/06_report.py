#!/usr/bin/env python
"""Step 6. Write RESULTS.md from the tables. Nothing in it is typed by hand.

    python scripts/06_report.py

Every number is read from results/tables/ at the moment this runs, so the
report cannot disagree with the tables. A section whose step has not been
run says so instead of being left out.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

from _common import load_config, md_table, out_dirs

import pandas as pd

from luad import ROOT
from luad.data import load_cohort
from luad.models import MODEL_LABELS

NOT_RUN = "_Not run yet._\n"
PART_TITLES = {"prediction": "Does a block add to clinical?",
               "integration": "Integration methods, nothing missing",
               "missing": "A block missing at test time",
               "control": "Pairing control (must be exactly zero)"}


def ci(v, lo, hi, signed=False) -> str:
    f = "+.3f" if signed else ".3f"
    return f"{v:{f}} ({lo:{f}}, {hi:{f}})"


def read(tables: Path, name: str) -> pd.DataFrame | None:
    p = tables / name
    return pd.read_csv(p) if p.exists() else None


def arms_wide(arms: pd.DataFrame, keep: list[str] | None = None) -> pd.DataFrame:
    """One row per arm, one column per model: Uno's C with its interval."""
    d = arms[arms["metric"] == "uno_c"].copy()
    d["cell"] = [ci(r.c_index, r.ci_low, r.ci_high) for r in d.itertuples()]
    w = d.pivot(index="arm", columns="model", values="cell")
    w = w.rename(columns=MODEL_LABELS)
    if keep:
        w = w.reindex([a for a in keep if a in w.index])
    return w.reset_index()


def contrast_table(con: pd.DataFrame) -> pd.DataFrame:
    d = con[con["metric"] == "uno_c"].copy()
    h = con[con["metric"] == "harrell_c"].set_index(["model", "a", "b"])["difference"]
    return pd.DataFrame({
        "contrast (a minus b)": [f"{r.a} minus {r.b}" for r in d.itertuples()],
        "model": d["model"].map(MODEL_LABELS).to_numpy(),
        "difference in Uno's C (95% CI)": [ci(r.difference, r.ci_low, r.ci_high, True)
                                           for r in d.itertuples()],
        "verdict": d["verdict"].to_numpy(),
        "Harrell's C": [f"{h.get((r.model, r.a, r.b), float('nan')):+.3f}" for r in d.itertuples()],
    })


def section_cv(tables: Path, cfg: dict, endpoint: str) -> str:
    arms, con = read(tables, f"arms_{endpoint}.csv"), read(tables, f"contrasts_{endpoint}.csv")
    if arms is None or con is None:
        return NOT_RUN
    out = []
    part1 = ["clinical", "rna", "mutation", "methylation", "clinical+rna",
             "clinical+mutation", "clinical+methylation", "early"]
    out += ["### Part 1. Prediction: clinical alone against clinical plus omics\n",
            "Uno's C with a 95% interval; `early` is all four blocks concatenated.\n",
            md_table(arms_wide(arms, part1)), ""]
    out += [md_table(contrast_table(con[con["part"] == "prediction"])), ""]

    out += ["### Part 2. Integration: three methods, with and without a missing block\n",
            "Rows are what the test patient is missing. `early` imputes a missing block "
            "with training means; `late` drops that block's model; `mofa` projects the "
            "patient onto the factors from the blocks that are present.\n"]
    d = arms[arms["metric"] == "uno_c"]
    for model, label in MODEL_LABELS.items():
        dm = d[d["model"] == model].set_index("arm")
        rows = []
        for sc in cfg["scenarios"]:
            row = {"missing at test": "nothing" if sc == "full" else sc.replace("_", " ")}
            for arm in ("early", "late", "mofa"):
                k = arm if sc == "full" else f"{arm}@{sc}"
                row[arm] = f"{dm.loc[k, 'c_index']:.3f}" if k in dm.index else "n/a"
            row["clinical model"] = f"{dm.loc['clinical', 'c_index']:.3f}"
            rows.append(row)
        out += [f"**{label}**\n", md_table(pd.DataFrame(rows)), ""]
    for part in ("integration", "missing"):
        out += [f"**{PART_TITLES[part]}**\n", md_table(contrast_table(con[con["part"] == part])), ""]

    real = con[(con["part"] != "control") & (con["metric"] == "uno_c")]
    counts = real["verdict"].value_counts()
    ctrl = con[con["part"] == "control"]
    zero = bool(((ctrl["difference"] == 0) & (ctrl["ci_low"] == 0) & (ctrl["ci_high"] == 0)).all())
    out += [f"**Tally.** {len(real)} pre-declared contrasts (the list in `config/config.yaml`, "
            f"each under two models): {int(counts.get('a better', 0))} favour a, "
            f"{int(counts.get('b better', 0))} favour b, "
            f"{int(counts.get('not resolved', 0))} not resolved. At 95% intervals about "
            f"{0.05 * len(real):.0f} would resolve by chance alone. "
            f"Pairing control exactly zero: **{'yes' if zero else 'NO'}**.\n"]
    return "\n".join(out)


def section_external(tables: Path) -> str:
    arms, con = read(tables, "external_arms.csv"), read(tables, "external_contrasts.csv")
    if arms is None or con is None:
        return NOT_RUN
    out = ["Models trained on TCGA and applied unchanged to GEO cohorts. GEO patients have "
           "expression and four clinical covariates only, so mutation and methylation are "
           "missing for real (`@geo`).\n"]
    d = arms[arms["metric"] == "uno_c"].copy()
    d["cell"] = [ci(r.c_index, r.ci_low, r.ci_high) for r in d.itertuples()]
    d["cohort"] = d["cohort"] + " " + d["endpoint"] + " (n=" + d["n"].astype(str) + ", " \
        + d["n_events"].astype(str) + " events)"
    for model, label in MODEL_LABELS.items():
        w = d[d["model"] == model].pivot(index="cohort", columns="arm", values="cell")
        w = w[[c for c in ["clinical", "clinical+rna", "early@geo", "late@geo", "mofa@geo"] if c in w]]
        out += [f"**{label}**\n", md_table(w.reset_index()), ""]
    c = con[con["metric"] == "uno_c"]
    t = (c.groupby(["a", "b"], sort=False)["verdict"].value_counts().unstack(fill_value=0)
          .reindex(columns=["a better", "not resolved", "b better"], fill_value=0).reset_index())
    t.insert(0, "contrast (a minus b)", t.pop("a") + " minus " + t.pop("b"))
    out += ["**Verdicts across cohorts, endpoints and models** (count of cohort-endpoint-model cells)\n",
            md_table(t), ""]
    return "\n".join(out)


def section_biology(tables: Path) -> str:
    factors = read(tables, "biology_factors.csv")
    if factors is None:
        return NOT_RUN
    out = ["### MOFA factors (whole cohort)\n",
           "`r2_*` is the share of a block's variance the factor explains; `cox_z` is its "
           "association with the endpoint (|z| > 2 is nominal); `r_*` are correlations.\n",
           md_table(factors, "+.2f"), ""]
    genes = read(tables, "biology_genes.csv")
    if genes is not None:
        lists = [c for c in genes.columns if genes[c].dtype == bool]
        out += ["### Gene lists\n",
                md_table(pd.DataFrame({"list": lists, "genes": [int(genes[c].sum()) for c in lists]})), ""]
    for name, title, note in [
        ("biology_overlap.csv", "Do the lists share genes?",
         "Hypergeometric test against all expressed genes."),
        ("biology_network_links.csv", "Are the lists linked in the protein network?",
         "Edges between two lists against the same count with labels shuffled."),
        ("biology_modules.csv", "Network modules", "Louvain modules of the STRING network."),
        ("biology_enrichment.csv", "Enrichment of the modules", "Top terms per module (STRING)."),
    ]:
        t = read(tables, name)
        out += [f"### {title}\n", note + "\n", md_table(t, ".3g") if t is not None else NOT_RUN, ""]
    cen = read(tables, "biology_network_centrality.csv")
    out += ["### Best-connected genes (Cytoscape network analysis)\n",
            "Topology statistics computed by Cytoscape itself in step 4b; top 15 by degree.\n",
            md_table(cen.head(15), ".3g") if cen is not None else NOT_RUN, ""]
    out += ["Figures drawn by Cytoscape: `results/figures/network_cytoscape_lists.png` (colour: "
            "gene list) and `network_cytoscape_coxz.png` (colour: Cox z, blue protective to red "
            "risk). To explore: open `results/network/network.cys` in Cytoscape.\n"]
    return "\n".join(out)


def section_checks(tables: Path) -> str:
    checks = read(tables, "checks.csv")
    if checks is None:
        return NOT_RUN
    out = [md_table(checks), ""]
    sweep = read(tables, "width_sweep.csv")
    if sweep is not None:
        out += ["**Width sweep** (fixed width instead of tuned; mean over folds)\n", md_table(sweep, "+.3f"), ""]
    return "\n".join(out)


def main() -> int:
    cfg = load_config()
    tables, _ = out_dirs(cfg)
    primary = cfg["endpoint"]["primary"].lower()
    sens = cfg["endpoint"]["sensitivity"].lower()
    c = load_cohort(cfg, primary, verbose=False)

    doc = [
        "# Results\n",
        f"_Generated by `scripts/06_report.py` on {date.today().isoformat()} from "
        "`results/tables/`. Do not edit by hand; rerun the script._\n",
        f"**Cohort.** {c.n} TCGA-LUAD patients with an endpoint and all four blocks "
        f"(clinical, expression, mutation, methylation). Primary endpoint {primary.upper()}, "
        f"censored at {cfg['endpoint']['admin_censor_months']} months: "
        f"{int(c.y['event'].sum())} events. "
        f"{cfg['cv']['outer_folds']}-fold cross-validation repeated {cfg['cv']['outer_repeats']} "
        "times; intervals from a paired patient-level bootstrap "
        f"({cfg['bootstrap']['n_resamples']} resamples).\n",
        f"## 1. Cross-validation, {primary.upper()} (primary)\n", section_cv(tables, cfg, primary),
        f"## 2. Cross-validation, {sens.upper()} (sensitivity)\n", section_cv(tables, cfg, sens),
        "## 3. External validation\n", section_external(tables),
        "## 4. Biology\n", section_biology(tables),
        "## 5. Checks\n", section_checks(tables),
    ]
    path = ROOT / "RESULTS.md"
    path.write_text("\n".join(doc))
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
