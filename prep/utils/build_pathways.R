#!/usr/bin/env Rscript
# build_pathways.R -- regenerate config/sanchez_vega_pathways.tsv
#
#   Rscript prep/utils/build_pathways.R
#
# Why this script exists
# ---------------------
# The pathway membership list was originally transcribed by hand from
# Sanchez-Vega et al. (2018) Table S3, and the transcription was wrong:
# 97 of the paper's 335 genes were missing and 16 genes that the paper
# does not list had been added. RTK-RAS -- the pathway altered in 81% of
# this LUAD cohort -- carried 44 of its 85 genes.
#
# Neither error is visible downstream. A missing gene silently weakens a
# pathway indicator; an invented one silently strengthens it. So the list
# is no longer maintained by hand: it is generated from the copy of Table
# S3 distributed with maftools, which is version-pinned in renv.lock and
# therefore reproducible from a clean checkout.
#
# maftools ships the table as inst/extdata/oncogenic_sig_patwhays.tsv
# (the filename typo is upstream's). Columns: Pathway, Gene, OG_TSG.
# OG_TSG is dropped -- the features here are "any non-silent mutation in
# any member gene", which does not use the oncogene/suppressor label.

suppressPackageStartupMessages(library(maftools))

OUT <- "config/sanchez_vega_pathways.tsv"

src <- system.file("extdata", "oncogenic_sig_patwhays.tsv", package = "maftools")
if (!nzchar(src) || !file.exists(src)) {
  stop("maftools does not ship oncogenic_sig_patwhays.tsv -- cannot rebuild ",
       OUT, ". Check the installed maftools version.")
}
ver <- as.character(packageVersion("maftools"))

ref <- read.delim(src, stringsAsFactors = FALSE)
stopifnot(all(c("Pathway", "Gene") %in% names(ref)))

# maftools' labels -> the snake_case keys the pipeline and Table 2b use.
# Mapped explicitly rather than lower-cased so that an upstream relabelling
# fails here instead of silently renaming a feature column.
KEYS <- c(Cell_Cycle = "cell_cycle", Hippo = "hippo", MYC = "myc",
          NOTCH = "notch", NRF2 = "nrf2", PI3K = "pi3k",
          `RTK-RAS` = "rtk_ras", `TGF-Beta` = "tgf_beta",
          TP53 = "tp53", WNT = "wnt")

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

out <- data.frame(pathway = unname(KEYS[ref$Pathway]),
                  gene    = ref$Gene,
                  stringsAsFactors = FALSE)
out <- unique(out)
out <- out[order(out$pathway, out$gene), ]

if (nrow(out) != 335L) {
  warning("expected 335 gene-pathway pairs, got ", nrow(out),
          " -- maftools ", ver, " may carry a revised list; ",
          "confirm against the paper before using these results")
}

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

con <- file(OUT, open = "wt")
writeLines(header, con)
write.table(out, con, sep = "\t", quote = FALSE, row.names = FALSE)
close(con)

cat("wrote ", OUT, ": ", nrow(out), " pairs, ",
    length(unique(out$pathway)), " pathways (maftools ", ver, ")\n", sep = "")
for (p in sort(unique(out$pathway))) {
  cat(sprintf("  %-12s %3d\n", p, sum(out$pathway == p)))
}
