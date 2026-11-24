#!/usr/bin/env Rscript
# =============================================================
# build_pathways.R -- write the list of pathway genes used by the mutation block
#
# What it is for
#   The mutation block has ten features of the form "does this patient have
#   a mutation in any gene of pathway X". This script writes the list that
#   says which gene belongs to which pathway. The ten pathways are the
#   curated oncogenic signalling pathways of Sanchez-Vega et al. (2018),
#   Table S3.
#
# Where it sits
#   Step 3 of 9 of `make prep`. prep/04_preprocess_mutation.py (step 5)
#   reads the file written here.
#
# Reads    the copy of that table that comes with the R package maftools.
#          Nothing is downloaded.
# Writes   config/sanchez_vega_pathways.tsv: two columns, pathway and gene,
#          under a comment header. The file is part of the repository, so
#          running this script overwrites a tracked file. The header records
#          the maftools version, so it changes if your version differs from
#          the 2.18.0 recorded in renv.lock.
#
# How to run (from the project folder)
#   Rscript prep/utils/build_pathways.R
#
# Why this script exists
# ---------------------
# The pathway membership list was originally transcribed by hand from
# Sanchez-Vega et al. (2018) Table S3, and the transcription was wrong:
# 97 of the paper's 335 genes were missing and 16 genes that the paper
# does not list had been added. RTK-RAS, the pathway most often altered in
# this cohort, carried 44 of its 85 genes.
#
# Neither error is visible downstream. A missing gene silently weakens a
# pathway indicator; an invented one silently strengthens it. So the list
# is no longer maintained by hand: it is generated from the copy of Table
# S3 distributed with maftools, whose version is recorded in renv.lock, so
# the same list can be produced again.
#
# maftools ships the table as inst/extdata/oncogenic_sig_patwhays.tsv
# (the filename typo is upstream's). Columns: Pathway, Gene, OG_TSG.
# OG_TSG says whether a gene is an oncogene or a tumour suppressor. It is
# dropped: the features here are "any non-silent mutation in any member
# gene" (a non-silent mutation is one that changes the protein), which
# does not use that label.
# =============================================================

# maftools is loaded only for the table that comes with it.
suppressPackageStartupMessages(library(maftools))

# The file to write.
OUT <- "config/sanchez_vega_pathways.tsv"

# Find the table inside the installed package, and stop if this version of
# maftools does not have it. The version is kept for the file header.
src <- system.file("extdata", "oncogenic_sig_patwhays.tsv", package = "maftools")
if (!nzchar(src) || !file.exists(src)) {
  stop("maftools does not ship oncogenic_sig_patwhays.tsv -- cannot rebuild ",
       OUT, ". Check the installed maftools version.")
}
ver <- as.character(packageVersion("maftools"))

# Read the table and check that the two columns needed are there.
ref <- read.delim(src, stringsAsFactors = FALSE)
stopifnot(all(c("Pathway", "Gene") %in% names(ref)))

# maftools writes the pathway names as "RTK-RAS", "TGF-Beta" and so on. The
# project uses lower-case names with underscores (rtk_ras, tgf_beta), which
# become the feature columns pw_rtk_ras, ... and the rows of
# results/tables/table2b_pathway_alteration.csv.
# Each name is mapped explicitly rather than lower-cased by a rule, so that
# a renamed pathway in a new maftools version stops the script here instead
# of silently renaming a feature column.
KEYS <- c(Cell_Cycle = "cell_cycle", Hippo = "hippo", MYC = "myc",
          NOTCH = "notch", NRF2 = "nrf2", PI3K = "pi3k",
          `RTK-RAS` = "rtk_ras", `TGF-Beta` = "tgf_beta",
          TP53 = "tp53", WNT = "wnt")

# Two checks: maftools must not contain a pathway that KEYS does not know,
# and must not lack one that KEYS expects.
unknown <- setdiff(unique(ref$Pathway), names(KEYS))
if (length(unknown)) {
  stop("unmapped pathway label(s) from maftools ", ver, ": ",
       paste(unknown, collapse = ", "),
       "\n  add them to KEYS -- do not lower-case blindly, the feature ",
       "names in Table 2b depend on these keys")
}
missing_keys <- setdiff(names(KEYS), unique(ref$Pathway))
if (length(missing_keys)) {
  stop("maftools ", ver, " no longer defines: ",
       paste(missing_keys, collapse = ", "))
}

# Build the two-column table, remove repeated rows and sort it, so that the
# file is identical every time it is generated.
out <- data.frame(pathway = unname(KEYS[ref$Pathway]),
                  gene    = ref$Gene,
                  stringsAsFactors = FALSE)
out <- unique(out)
out <- out[order(out$pathway, out$gene), ]

# The table is expected to have 335 gene-pathway pairs. A different number
# gives a warning, not an error: the file is still written, and should be
# compared with the paper before it is used.
if (nrow(out) != 335L) {
  warning("expected 335 gene-pathway pairs, got ", nrow(out),
          " -- maftools ", ver, " may carry a revised list; ",
          "confirm against the paper before using these results")
}

# The comment lines written at the top of the output file. They say where
# the list comes from and that the file must not be edited by hand.
header <- c(
  "# Ten curated oncogenic signalling pathways",
  "# Sanchez-Vega et al. (2018) \"Oncogenic Signaling Pathways in The Cancer",
  "# Genome Atlas\", Cell 173:321-337, Table S3.",
  "#",
  "# ---------------------------------------------------------------------",
  "# GENERATED FILE -- do not edit by hand.",
  "#   Rscript prep/utils/build_pathways.R",
  "#",
  paste0("# Source: Table S3 as distributed with maftools ", ver,
         " (inst/extdata/"),
  "# oncogenic_sig_patwhays.tsv), version-pinned in renv.lock. An earlier",
  "# hand-transcribed version of this file was missing 97 of the 335 genes",
  "# and had 16 genes the paper does not list; RTK-RAS held 44 of its 85.",
  "# That is why the list is generated rather than curated here.",
  "# ---------------------------------------------------------------------",
  "#",
  "# Symbols are HGNC and are matched against MC3's Hugo_Symbol column.",
  "# A gene may appear in more than one pathway; that is how the paper",
  "# defines them and the features are not mutually exclusive.",
  paste0("#"),
  paste0("# ", nrow(out), " gene-pathway pairs across ",
         length(unique(out$pathway)), " pathways.")
)

# Write the header, then the table, tab-separated.
con <- file(OUT, open = "wt")
writeLines(header, con)
write.table(out, con, sep = "\t", quote = FALSE, row.names = FALSE)
close(con)

# Print what was written and the number of genes per pathway.
cat("wrote ", OUT, ": ", nrow(out), " pairs, ",
    length(unique(out$pathway)), " pathways (maftools ", ver, ")\n", sep = "")
for (p in sort(unique(out$pathway))) {
  cat(sprintf("  %-12s %3d\n", p, sum(out$pathway == p)))
}
