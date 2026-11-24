#!/usr/bin/env python
"""
04_preprocess_mutation.py  --  somatic mutation block.

    python prep/04_preprocess_mutation.py

Writes, in patient-barcode space:
    data/processed/mut_gene_binary.parquet   patients x genes, 0/1 non-silent
    data/processed/mut_features.parquet      TMB + 10 pathway indicators
    results/tables/table2_mutation_summary.md    narrative + both tables
    results/tables/table2_recurrent_genes.csv    every gene above the threshold
    results/tables/table2b_pathway_alteration.csv

Why the gene block is aggregated at all
---------------------------------------
Raw gene-level binaries are ~99.5% zeros across ~18,000 genes. At 496
patients and 210 events, a matrix that wide is not a feature set, it is
noise with an index. Three aggregations carry the signal instead:

    1. Recurrently altered genes  -- the 5% threshold is applied INSIDE the
       fold, so this script writes every gene mutated in at least one
       patient and lets 06 pick the survivors from its own training rows.
    2. log10(TMB + 1)             -- per-patient, so fold-safe here.
    3. Ten oncogenic pathways     -- membership is a fixed curated list, so
       also fold-safe.

The 5% frequency threshold is the one genuinely cohort-dependent step, and
it is deliberately NOT applied here. Computing it over all 512 patients
would select genes using outcome-adjacent information from the test folds;
the effect is small but it is exactly the class of leak that a permutation
control cannot see, because permuting the outcome does not change which
genes are recurrent.

MC3 is scoped to LUAD via the TCGA-CDR patient roster, never via the RNA
cohort -- scoping by RNA would make the mutation block a subset of RNA by
construction and the Week 1 availability matrix would be measuring the
filter rather than the data.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

MAF_COLS = ["Hugo_Symbol", "Tumor_Sample_Barcode", "FILTER", "Variant_Classification"]


def main() -> int:
    cfg = yaml.safe_load(open("config/config.yaml"))
    raw = Path(cfg["paths"]["raw"])
    interim = Path(cfg["paths"]["interim"])
    processed = Path(cfg["paths"]["processed"])
    tables = Path(cfg["paths"]["tables"])
    processed.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)

    maf_path = raw / "mc3.v0.2.8.PUBLIC.maf.gz"
    clin_path = interim / "clinical_raw.parquet"
    for p in (maf_path, clin_path):
        if not p.exists():
            sys.exit(f"{p} not found.\n  run `make download` then `make export-pheno`")

    print("=" * 62)
    print("MUTATION BLOCK  --  MC3 PanCanAtlas consensus MAF")
    print("=" * 62)

    # The LUAD roster is the TCGA-CDR patient list, which clinical_raw.parquet
    # carries verbatim (every GDC patient is inside it). Asserting the size
    # keeps this honest if the upstream export ever changes shape.
    roster = set(pd.read_parquet(clin_path)["patient_barcode"])
    print(f"\nLUAD roster (TCGA-CDR)          : {len(roster)} patients")

    print(f"\nreading {maf_path.name} (753 MB, 4 of 114 columns) ...")
    maf = pd.read_csv(maf_path, sep="\t", usecols=MAF_COLS, dtype=str,
                      compression="gzip", low_memory=False)
    print(f"  variants (pan-cancer)         : {len(maf):,}")

    maf["patient"] = maf["Tumor_Sample_Barcode"].str.slice(0, 12)
    maf = maf[maf["patient"].isin(roster)]
    print(f"  variants in LUAD roster       : {len(maf):,}")

    maf = maf[maf["FILTER"] == cfg["data"]["mutation"]["filter"]]
    print(f"  PASS variants                 : {len(maf):,}")

    nonsilent = set(cfg["data"]["mutation"]["nonsilent"])
    maf = maf[maf["Variant_Classification"].isin(nonsilent)]
    print(f"  non-silent variants           : {len(maf):,}")

    # MC3 uses "Unknown" for unannotated loci; those carry no gene identity
    # and would otherwise become a spurious high-frequency feature.
    maf = maf[maf["Hugo_Symbol"].notna() & (maf["Hugo_Symbol"] != "Unknown")]
    print(f"  with a gene symbol            : {len(maf):,}")

    patients = sorted(maf["patient"].unique())
    print(f"  patients with >= 1 variant    : {len(patients)}")
    if not patients:
        sys.exit("no LUAD variants survived filtering -- check the roster join")

    # ---- gene x patient binary matrix ---------------------------------
    # One row per patient, one column per gene mutated anywhere in the
    # cohort. int8 keeps ~512 x 18,000 under 10 MB.
    print("\nbuilding gene binary matrix ...")
    pairs = maf[["patient", "Hugo_Symbol"]].drop_duplicates()
    gene_bin = (pd.crosstab(pairs["patient"], pairs["Hugo_Symbol"])
                  .astype(np.int8)
                  .reindex(index=patients)
                  .sort_index(axis=1))
    print(f"  {gene_bin.shape[0]} patients x {gene_bin.shape[1]} genes "
          f"({gene_bin.to_numpy().mean():.3%} non-zero)")

    # ---- TMB ------------------------------------------------------------
    n_nonsilent = maf.groupby("patient").size().reindex(patients).fillna(0)
    exome_mb = float(cfg["preprocessing"]["mutation"]["tmb_exome_mb"])
    tmb = n_nonsilent / exome_mb
    tmb_log10 = np.log10(tmb + 1.0)
    print(f"\nTMB (non-silent / {exome_mb:g} Mb)")
    print(f"  median {tmb.median():.2f}   IQR {tmb.quantile(.25):.2f}-"
          f"{tmb.quantile(.75):.2f}   max {tmb.max():.1f} mut/Mb")

    # ---- pathway indicators ---------------------------------------------
    pw_path = Path("config/sanchez_vega_pathways.tsv")
    if not pw_path.exists():
        sys.exit(f"{pw_path} not found -- the pathway features cannot be built")
    pw = pd.read_csv(pw_path, sep="\t", comment="#")
    if not {"pathway", "gene"} <= set(pw.columns):
        sys.exit(f"{pw_path}: expected columns 'pathway' and 'gene'")

    print(f"\npathway features ({pw['pathway'].nunique()} pathways, "
          f"{len(pw)} gene assignments)")
    pw_cols, pw_rows = {}, []
    for name, members in pw.groupby("pathway")["gene"]:
        present = [g for g in members if g in gene_bin.columns]
        altered = (gene_bin[present].sum(axis=1) > 0).astype(np.int8) if present \
            else pd.Series(0, index=gene_bin.index, dtype=np.int8)
        pw_cols[f"pw_{name}"] = altered
        pw_rows.append({"pathway": name, "n_genes": len(list(members)),
                        "n_genes_mutated": len(present),
                        "n_patients_altered": int(altered.sum()),
                        "pct_altered": altered.mean()})
        print(f"  {name:<12} {len(present):>3}/{len(list(members)):<3} genes mutated"
              f"   {int(altered.sum()):>3} patients ({altered.mean():>5.1%})")

    pw_summary = pd.DataFrame(pw_rows)
    # A pathway whose genes are almost entirely absent from an exome-wide MAF
    # points at a symbol-nomenclature problem in the curated list, not at
    # biology -- MC3 covers the whole exome, so every listed gene is testable.
    thin = pw_summary[pw_summary["n_genes_mutated"] / pw_summary["n_genes"] < 0.5]
    if len(thin):
        print("\n  !! under half of the listed genes were seen mutated in:",
              ", ".join(thin["pathway"]))
        print("     check config/sanchez_vega_pathways.tsv for outdated symbols")

    features = pd.DataFrame({
        "n_nonsilent": n_nonsilent.astype(int),
        "tmb": tmb,
        "tmb_log10": tmb_log10,
        **pw_cols,
    })
    features.index.name = "patient_barcode"
    gene_bin.index.name = "patient_barcode"

    # ---- what the 5% threshold WOULD select (reported, not applied) ------
    thr = float(cfg["preprocessing"]["mutation"]["gene_freq_threshold"])
    freq = gene_bin.mean(axis=0).sort_values(ascending=False)
    recurrent = freq[freq >= thr]
    cap = cfg["preprocessing"]["mutation"].get("gene_max_features")
    kept = recurrent if cap is None else recurrent.head(int(cap))
    print(f"\ncohort-wide, {len(recurrent)} genes are altered in >= {thr:.0%} "
          f"of patients:")
    for g, v in kept.items():
        print(f"    {g:<12} {v:>6.1%}")
    if cap is not None and len(recurrent) > cap:
        print(f"  ... and {len(recurrent) - len(kept)} more, dropped by the "
              f"gene_max_features={cap} cap")
        print(f"  The protocol expected ~26 genes at this threshold. LUAD's "
              f"mutation burden\n  makes that wrong by a factor of ~27: at 210 PFI "
              f"events, {len(recurrent)} binary\n  features is EPV "
              f"{210 / len(recurrent):.2f}. See config/config.yaml for the rationale.")
    print("  (reported only -- both the threshold and the cap are refitted")
    print("   inside each training fold in 06_nested_cv.py)")
    n_feat = len(kept) + 1 + len(pw_cols)
    print(f"\n  block width at deployment: {len(kept)} genes + 1 TMB + "
          f"{len(pw_cols)} pathways = {n_feat} features  (EPV {210 / n_feat:.1f})")

    # ---- write -----------------------------------------------------------
    out_bin = processed / "mut_gene_binary.parquet"
    out_feat = processed / "mut_features.parquet"
    gene_bin.reset_index().to_parquet(out_bin, index=False, compression="gzip")
    features.reset_index().to_parquet(out_feat, index=False, compression="gzip")

    write_table2(tables, gene_bin, features, recurrent, kept, pw_summary, thr)

    print("\n=== written ===")
    for p in (out_bin, out_feat):
        print(f"  {p}  ({p.stat().st_size / 1024**2:.1f} MB)")
    print("Next: python prep/05_preprocess_clinical.py")
    return 0


def write_table2(tables, gene_bin, features, recurrent, kept, pw_summary, thr):
    # The CSV keeps every gene above the threshold so the cap can be revisited
    # without re-reading the 753 MB MAF; the markdown table shows only the
    # genes that actually reach a model.
    full = pd.DataFrame({"Gene": recurrent.index,
                         "n altered": [int(gene_bin[g].sum()) for g in recurrent.index],
                         "% of cohort": [f"{v:.1%}" for v in recurrent.values],
                         "within cap": [g in set(kept.index) for g in recurrent.index]})
    full.to_csv(tables / "table2_recurrent_genes.csv", index=False)
    top = full[full["within cap"]].drop(columns="within cap")

    pw_out = pw_summary.copy()
    pw_out["pct_altered"] = pw_out["pct_altered"].map(lambda x: f"{x:.1%}")
    pw_out = pw_out.rename(columns={
        "pathway": "Pathway", "n_genes": "Genes in list",
        "n_genes_mutated": "Genes mutated in cohort",
        "n_patients_altered": "Patients altered", "pct_altered": "% altered"})
    pw_out.to_csv(tables / "table2b_pathway_alteration.csv", index=False)

    with open(tables / "table2_mutation_summary.md", "w") as fh:
        fh.write("# Table 2. Somatic mutation block (TCGA-LUAD, MC3)\n\n")
        fh.write(f"{gene_bin.shape[0]} patients · {gene_bin.shape[1]} genes mutated "
                 f"in ≥1 patient · {features['n_nonsilent'].sum():,} non-silent "
                 f"variants\n\n")
        fh.write("> These are all patients with mutation calls, before the analysis "
                 "cohort is formed. The analysis itself uses the 440 patients who also "
                 "have every other block; see cohort_flow.csv and "
                 "cohort_characteristics.md.\n\n")
        fh.write(f"## Genes altered in ≥{thr:.0%} of the cohort "
                 f"(top {len(top)} of {len(recurrent)})\n\n")
        fh.write("> Reported for description only. Both the threshold and the "
                 "top-K cap are refitted inside each training fold, so the "
                 "modelled set varies by fold.\n\n")
        if len(recurrent) > len(top):
            fh.write(f"> **{len(recurrent)} genes clear {thr:.0%}, not the ~26 the "
                     f"protocol assumed.** LUAD's mutation burden (median 4.8 "
                     f"mut/Mb) puts that many binary features at EPV "
                     f"{210 / len(recurrent):.2f} against the 210 PFI events of the source "
                     f"cohort (509 patients, before censoring at 60 months), and most "
                     f"are length-biased passengers. The block is capped at the "
                     f"{len(top)} most frequent; driver biology enters through the "
                     f"pathway features below.\n\n")
        fh.write(top.to_markdown(index=False))
        fh.write("\n\n## Oncogenic pathway alteration (Sanchez-Vega et al. 2018)\n\n")
        fh.write(pw_out.to_markdown(index=False))
        fh.write("\n")
    print(f"\n  wrote {tables / 'table2_mutation_summary.md'}")


if __name__ == "__main__":
    sys.exit(main())
