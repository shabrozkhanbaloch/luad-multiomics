#!/usr/bin/env python
"""
04_preprocess_mutation.py -- the somatic mutation block.

What it is for
    Turns the MC3 mutation file, which covers every TCGA cancer type, into
    the two mutation tables the analysis reads for the LUAD patients:
    which genes are mutated in which patient, and, per patient, the tumour
    mutation burden and ten pathway indicators.

Words used below
    - A somatic mutation is one found in the tumour and not in the
      patient's normal tissue.
    - A MAF file (Mutation Annotation Format) is a tab-separated table
      with one row per mutation found in one tumour.
    - A non-silent mutation changes the protein (missense, nonsense,
      frameshift, ...; the list is in config/config.yaml). A silent one
      changes the DNA but not the protein and is not counted here.
    - TMB, tumour mutation burden: the number of non-silent mutations of
      a tumour per megabase (million bases) of sequenced DNA.
    - EPV, events per variable: the number of outcome events divided by
      the number of features. A low EPV means a model has too many
      features for the outcomes it is fitted on.

Where it sits
    Step 6 of 9 of `make prep`, after `make download`.

Reads
    data/raw/mc3.v0.2.8.PUBLIC.maf.gz   the MC3 file (753 MB)
    data/interim/clinical_raw.parquet   used only as the list of LUAD
                                        patients. prep/utils/export_clinical.R
                                        writes it, so that script must have
                                        run before this one.
    config/sanchez_vega_pathways.tsv    which gene is in which pathway
    config/config.yaml                  the filter, the variant types, the
                                        exome size, the 5% threshold and cap

Writes, with one row per patient (12-character patient barcode)
    data/processed/mut_gene_binary.parquet   patients x genes, 1 if the gene
                                             has a non-silent mutation, else 0
    data/processed/mut_features.parquet      number of non-silent mutations,
                                             TMB, log10(TMB + 1), and ten
                                             pathway indicators (pw_*)
    results/tables/table2_mutation_summary.md     a summary with both tables
    results/tables/table2_recurrent_genes.csv     every gene above the threshold
    results/tables/table2b_pathway_alteration.csv patients altered per pathway

How to run (from the project folder, with the project's Python)
    .venv/bin/python prep/04_preprocess_mutation.py

The README states that this step was rerun in this repository and
reproduced the files in data/processed exactly.

Why the gene block is aggregated at all
---------------------------------------
Raw gene-level 0/1 values are ~99.5% zeros across ~18,000 genes. With
about 500 patients and 210 events, a table that wide is almost all noise.
Three summaries carry the signal instead:

    1. Recurrently altered genes: the genes mutated in at least 5% of
       patients. The threshold is applied INSIDE each cross-validation
       fold, so this script writes every gene mutated in at least one
       patient and lets the analysis pick the recurrent ones from its own
       training patients (src/luad/transforms.py).
    2. log10(TMB + 1): computed for each patient from that patient's own
       mutations, so it is safe to compute here.
    3. Ten oncogenic pathways: 1 if any gene of the pathway is mutated.
       Which gene belongs to which pathway is a fixed published list, so
       this is also safe to compute here.

The 5% frequency threshold is the one step that depends on the group of
patients it is computed on, and it is deliberately NOT applied here.
Computed over all 512 patients, it would choose the genes using patients
who are later held out to test the models. The effect is small, but it is
exactly the kind of leak that the shuffled-outcome check (`make permute`)
cannot see, because shuffling the outcome does not change which genes are
recurrent.

The LUAD patients are taken from the TCGA-CDR patient list, never from the
patients who have RNA-seq. Selecting by RNA-seq would make the mutation
block a subset of the expression block by construction, and the number of
patients with mutation data would describe that filter rather than the
data.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# The four columns of the MAF file that are read: the gene symbol, the
# barcode of the tumour sample, the quality flag (PASS means no flag was
# raised) and the kind of change.
MAF_COLS = ["Hugo_Symbol", "Tumor_Sample_Barcode", "FILTER", "Variant_Classification"]


def main() -> int:
    """Build the mutation tables and the summary tables; return 0 on success."""
    # Folders from the configuration; create the two output folders.
    cfg = yaml.safe_load(open("config/config.yaml"))
    raw = Path(cfg["paths"]["raw"])
    interim = Path(cfg["paths"]["interim"])
    processed = Path(cfg["paths"]["processed"])
    tables = Path(cfg["paths"]["tables"])
    processed.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)

    # Both inputs must exist. The message below names `make export-pheno`,
    # a target of the earlier project that does not exist here; the script
    # it stands for is prep/utils/export_clinical.R.
    maf_path = raw / "mc3.v0.2.8.PUBLIC.maf.gz"
    clin_path = interim / "clinical_raw.parquet"
    for p in (maf_path, clin_path):
        if not p.exists():
            sys.exit(f"{p} not found.\n  run `make download`, then `Rscript prep/utils/export_clinical.R`")

    print("=" * 62)
    print("MUTATION BLOCK  --  MC3 PanCanAtlas consensus MAF")
    print("=" * 62)

    # The list of LUAD patients ("roster") is the TCGA-CDR patient list.
    # clinical_raw.parquet contains every one of those patients (and every
    # GDC patient is among them). Its size is printed, so that a change in
    # the upstream export is noticed.
    roster = set(pd.read_parquet(clin_path)["patient_barcode"])
    print(f"\nLUAD roster (TCGA-CDR)          : {len(roster)} patients")

    # Read the four columns as text. The file holds every TCGA cancer type.
    print(f"\nreading {maf_path.name} (753 MB, 4 of 114 columns) ...")
    maf = pd.read_csv(maf_path, sep="\t", usecols=MAF_COLS, dtype=str,
                      compression="gzip", low_memory=False)
    print(f"  variants (pan-cancer)         : {len(maf):,}")

    # The first 12 characters of a TCGA sample barcode identify the patient.
    # Keep the rows of the LUAD patients. Each filter below prints how many
    # rows remain, so the log shows what every step removed.
    maf["patient"] = maf["Tumor_Sample_Barcode"].str.slice(0, 12)
    maf = maf[maf["patient"].isin(roster)]
    print(f"  variants in LUAD roster       : {len(maf):,}")

    # Keep mutation calls that passed every quality filter (FILTER == PASS).
    maf = maf[maf["FILTER"] == cfg["data"]["mutation"]["filter"]]
    print(f"  PASS variants                 : {len(maf):,}")

    # Keep non-silent mutations: the kinds of change listed in the config.
    nonsilent = set(cfg["data"]["mutation"]["nonsilent"])
    maf = maf[maf["Variant_Classification"].isin(nonsilent)]
    print(f"  non-silent variants           : {len(maf):,}")

    # MC3 writes "Unknown" where a position has no gene. Those rows belong to
    # no gene, and together they would look like one very often mutated
    # gene, so they are removed.
    maf = maf[maf["Hugo_Symbol"].notna() & (maf["Hugo_Symbol"] != "Unknown")]
    print(f"  with a gene symbol            : {len(maf):,}")

    # The patients of this block: everyone with at least one mutation left.
    patients = sorted(maf["patient"].unique())
    print(f"  patients with >= 1 variant    : {len(patients)}")
    if not patients:
        sys.exit("no LUAD variants survived filtering -- check the roster join")

    # ---- gene x patient binary matrix ---------------------------------
    # One row per patient, one column per gene mutated anywhere in the
    # cohort: 1 if the patient has a non-silent mutation in the gene, else
    # 0. Several mutations in one gene count once (drop_duplicates).
    # int8, a one-byte integer, keeps ~512 x 18,000 under 10 MB.
    print("\nbuilding gene binary matrix ...")
    pairs = maf[["patient", "Hugo_Symbol"]].drop_duplicates()
    gene_bin = (pd.crosstab(pairs["patient"], pairs["Hugo_Symbol"])
                  .astype(np.int8)
                  .reindex(index=patients)
                  .sort_index(axis=1))
    print(f"  {gene_bin.shape[0]} patients x {gene_bin.shape[1]} genes "
          f"({gene_bin.to_numpy().mean():.3%} non-zero)")

    # ---- TMB ------------------------------------------------------------
    # Tumour mutation burden: the patient's number of non-silent mutations
    # divided by the size of the sequenced exome in megabases (tmb_exome_mb
    # in the config, 38). The analysis uses log10(TMB + 1): a few tumours
    # have very many mutations, and the logarithm stops them dominating.
    # The "+ 1" keeps a burden of zero at zero.
    n_nonsilent = maf.groupby("patient").size().reindex(patients).fillna(0)
    exome_mb = float(cfg["preprocessing"]["mutation"]["tmb_exome_mb"])
    tmb = n_nonsilent / exome_mb
    tmb_log10 = np.log10(tmb + 1.0)
    print(f"\nTMB (non-silent / {exome_mb:g} Mb)")
    print(f"  median {tmb.median():.2f}   IQR {tmb.quantile(.25):.2f}-"
          f"{tmb.quantile(.75):.2f}   max {tmb.max():.1f} mut/Mb")

    # ---- pathway indicators ---------------------------------------------
    # For each of the ten pathways: 1 if the patient has a non-silent
    # mutation in at least one gene of the pathway, else 0. The gene list is
    # config/sanchez_vega_pathways.tsv (written by
    # prep/utils/build_pathways.R). A gene of the list that is never mutated
    # in the cohort has no column in the gene matrix and is skipped.
    # `pw_rows` collects one summary row per pathway for the tables.
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
    # A warning, not a filter. MC3 covers the whole exome, so every listed
    # gene could have been found mutated. If fewer than half of a pathway's
    # genes are ever seen, the likely cause is an outdated gene symbol in
    # the list, not biology.
    thin = pw_summary[pw_summary["n_genes_mutated"] / pw_summary["n_genes"] < 0.5]
    if len(thin):
        print("\n  !! under half of the listed genes were seen mutated in:",
              ", ".join(thin["pathway"]))
        print("     check config/sanchez_vega_pathways.tsv for outdated symbols")

    # The per-patient feature table: mutation count, TMB, log TMB and the
    # ten pathway columns. Both tables are indexed by patient barcode.
    features = pd.DataFrame({
        "n_nonsilent": n_nonsilent.astype(int),
        "tmb": tmb,
        "tmb_log10": tmb_log10,
        **pw_cols,
    })
    features.index.name = "patient_barcode"
    gene_bin.index.name = "patient_barcode"

    # ---- what the 5% threshold WOULD select (reported, not applied) ------
    # For description only: the genes mutated in at least 5% of all
    # patients, and the most frequent of them up to the cap
    # (gene_max_features in the config). The analysis repeats both choices
    # on the training patients of each fold.
    # Some of the printed text was written for the earlier project: "the
    # protocol" is that project's plan, 06_nested_cv.py is its
    # cross-validation script (here: scripts/01_cv.py), and config.yaml
    # here does not contain the rationale that the message points to.
    # 210 is the number of PFI events in the source cohort, written into
    # the code as a number, not read from the data.
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
        print(f"  That is far more genes than the events can support: at the 210 PFI "
              f"events\n  of the source cohort, {len(recurrent)} yes/no features is "
              f"{210 / len(recurrent):.2f} events per feature.\n  This is why the block "
              f"is capped (gene_max_features in config/config.yaml).")
    print("  (reported only -- both the threshold and the cap are refitted")
    print("   inside each training fold, in src/luad/transforms.py)")
    n_feat = len(kept) + 1 + len(pw_cols)
    print(f"\n  block width at deployment: {len(kept)} genes + 1 TMB + "
          f"{len(pw_cols)} pathways = {n_feat} features  (EPV {210 / n_feat:.1f})")

    # ---- write -----------------------------------------------------------
    # reset_index() turns the patient barcode from the row label into an
    # ordinary first column, so that R and Python both read it.
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
    """Write the three descriptive tables of the mutation block.

    The numbers describe all patients with mutation calls. They are for
    the reader; no model reads these tables.
    """
    # The CSV keeps every gene above the threshold, with a column saying
    # whether it is within the cap, so that the cap can be reconsidered
    # without reading the 753 MB MAF file again. The markdown table shows
    # only the genes within the cap.
    full = pd.DataFrame({"Gene": recurrent.index,
                         "n altered": [int(gene_bin[g].sum()) for g in recurrent.index],
                         "% of cohort": [f"{v:.1%}" for v in recurrent.values],
                         "within cap": [g in set(kept.index) for g in recurrent.index]})
    full.to_csv(tables / "table2_recurrent_genes.csv", index=False)
    top = full[full["within cap"]].drop(columns="within cap")

    # The pathway table, with readable column names and percentages.
    pw_out = pw_summary.copy()
    pw_out["pct_altered"] = pw_out["pct_altered"].map(lambda x: f"{x:.1%}")
    pw_out = pw_out.rename(columns={
        "pathway": "Pathway", "n_genes": "Genes in list",
        "n_genes_mutated": "Genes mutated in cohort",
        "n_patients_altered": "Patients altered", "pct_altered": "% altered"})
    pw_out.to_csv(tables / "table2b_pathway_alteration.csv", index=False)

    # The summary as a markdown file: a headline, both tables, and notes
    # that say what the numbers do and do not describe.
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
