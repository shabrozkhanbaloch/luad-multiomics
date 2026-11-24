#!/usr/bin/env Rscript
# =============================================================
# 03_preprocess_rna.R  --  RNA-seq block, fold-safe steps only
#
#   Rscript prep/03_preprocess_rna.R
#
# Writes, in patient-barcode space:
#   data/processed/rna_counts.parquet     raw STAR counts, genes x patients
#   data/processed/rna_rank.parquet       within-sample ranks in [0,1]  <- PRIMARY
#   data/processed/rna_gene_meta.parquet  gene_id, symbol, biotype, chromosome
#
# WHAT THIS SCRIPT DELIBERATELY DOES NOT DO
# -----------------------------------------
# filterByExpr, the top-variance prefilter, VST dispersion estimation and
# per-cohort z-scoring are all COHORT-DEPENDENT: their parameters are
# functions of the samples they see. Running them here, once, over all 517
# patients would leak test-fold information into every training fold and
# the label-permutation control in 07 would not be able to detect it --
# permuting the outcome leaves an unsupervised leak completely invisible.
# They are therefore implemented in prep/utils/foldwise.py and refitted
# inside each inner training partition.
#
# Only two things happen here, and both are safe by construction:
#
#   1. Annotation-driven selection (protein-coding, primary tumour, one
#      aliquot per patient). Depends on GENCODE and on the barcode, never
#      on the expression values.
#   2. The within-sample rank transform. Each patient's ranks are computed
#      across that patient's own transcriptome and are unchanged by which
#      other patients are in the fold -- the property that makes rank the
#      primary representation rather than a convenience.
#
# The single exception is dropping genes with zero counts in every one of
# the 517 patients. Such a gene is necessarily all-zero in any subset, so
# no fold can learn anything from its removal.
# =============================================================

suppressPackageStartupMessages({
  library(yaml); library(arrow); library(data.table)
  library(SummarizedExperiment)
})

cfg       <- yaml::read_yaml("config/config.yaml")
RAW       <- cfg$paths$raw
PROCESSED <- cfg$paths$processed
set.seed(cfg$project$seed)
dir.create(PROCESSED, recursive = TRUE, showWarnings = FALSE)

log_msg <- function(...) cat(format(Sys.time(), "[%H:%M:%S] "), ..., "\n", sep = "")

patient_of     <- function(bc) substr(bc, 1, 12)
sample_type_of <- function(bc) substr(bc, 14, 15)
vial_of        <- function(bc) substr(bc, 16, 16)
plate_of       <- function(bc) substr(bc, 22, 25)

f <- file.path(RAW, "tcga_luad_rnaseq_se.rds")
if (!file.exists(f)) stop("missing ", f, "\n  run `make download` first")

log_msg("=== loading SummarizedExperiment ===")
se <- readRDS(f)
log_msg("  ", nrow(se), " genes x ", ncol(se), " aliquots")

# STAR emits several assays; `unstranded` holds the raw counts the TCGA
# harmonised pipeline reports. Everything downstream assumes integers.
if (!("unstranded" %in% assayNames(se)))
  stop("no `unstranded` assay; present: ", paste(assayNames(se), collapse = ", "))

# ---------------------------------------------------------------
# Sample selection -- must match prep/utils/export_barcodes.R exactly,
# or Table 1 and the design matrix describe different cohorts.
# ---------------------------------------------------------------
log_msg("=== sample selection ===")
bc   <- colnames(se)
keep <- sample_type_of(bc) %in% cfg$data$tcga$sample_types
log_msg("  primary solid tumour (type 01): ", sum(keep),
        "  (dropped ", sum(!keep), ")")
se <- se[, keep]

dt <- data.table(barcode = colnames(se), patient = patient_of(colnames(se)),
                 vial = vial_of(colnames(se)), plate = plate_of(colnames(se)))
n_dup <- sum(duplicated(dt$patient))
setorder(dt, patient, vial, plate)
dt <- dt[, .SD[1], by = patient]
se <- se[, dt$barcode]
colnames(se) <- dt$patient
log_msg("  one aliquot per patient: ", ncol(se), "  (resolved ", n_dup, " duplicates)")

if (anyDuplicated(colnames(se))) stop("duplicate patient barcodes survived deduplication")

# ---------------------------------------------------------------
# Gene selection -- annotation only
# ---------------------------------------------------------------
log_msg("=== gene selection ===")
rd <- as.data.table(rowData(se))
rd[, gene_id := rownames(se)]

# The biotype and the output prefix are overridable from the command line
# so that a second block can be built from the SAME assay without touching
# the protein-coding path. Defaults reproduce the committed run exactly:
#
#   Rscript prep/03_preprocess_rna.R                        -> rna_*.parquet
#   Rscript prep/03_preprocess_rna.R --biotype lncRNA --prefix lnc
#                                                          -> lnc_*.parquet
#
# Ellen et al. (2023) report the one credible multi-omics gain in LUAD from
# lncRNA, and the STAR-counts object already carries 16,901 of them; they
# were filtered out here, not absent from the download.
args   <- commandArgs(trailingOnly = TRUE)
argval <- function(flag, default) {
  i <- match(flag, args)
  if (is.na(i) || i == length(args)) default else args[[i + 1L]]
}
want   <- argval("--biotype", cfg$preprocessing$expression$gene_biotype)
prefix <- argval("--prefix",  "rna")

if (!(want %in% rd$gene_type))
  stop("biotype '", want, "' absent from rowData$gene_type; present: ",
       paste(head(sort(unique(rd$gene_type)), 20), collapse = ", "))

is_pc <- rd$gene_type == want
log_msg("  ", want, ": ", sum(is_pc), "  (dropped ", sum(!is_pc), " other biotypes)")
se <- se[is_pc, ]
rd <- rd[is_pc]

counts <- assay(se, "unstranded")
storage.mode(counts) <- "integer"

# Structurally silent genes. All-zero across the full cohort implies
# all-zero in every possible training subset, so this cannot leak.
nz <- rowSums(counts) > 0
log_msg("  non-zero in >= 1 patient: ", sum(nz), "  (dropped ", sum(!nz),
        " all-zero genes)")
counts <- counts[nz, , drop = FALSE]
rd     <- rd[nz]

stopifnot(identical(rownames(counts), rd$gene_id))

# ---------------------------------------------------------------
# Within-sample rank transform -- the primary representation.
#
# Average ranks per patient, divided by the number of genes, so each
# column lands in (0, 1]. Library size, platform scale and per-cohort
# distributional shape all drop out, which is what makes a model fitted
# on TCGA RNA-seq evaluable on GEO microarrays at all.
#
# The many tied zeros collapse to one shared mid-low rank; that is the
# intended behaviour, not a defect -- it encodes "not detected" as a
# single level instead of inventing an order within it.
# ---------------------------------------------------------------
log_msg("=== within-sample rank transform ===")
n_genes <- nrow(counts)
ranks <- matrix(0, nrow = n_genes, ncol = ncol(counts),
                dimnames = dimnames(counts))
for (j in seq_len(ncol(counts))) {
  ranks[, j] <- rank(counts[, j], ties.method = "average") / n_genes
}
storage.mode(ranks) <- "double"

stopifnot(all(ranks > 0), all(ranks <= 1))
log_msg("  range: [", sprintf("%.4f", min(ranks)), ", ",
        sprintf("%.4f", max(ranks)), "]  over ", n_genes, " genes")
# Every column is a permutation of the same rank multiset, so all column
# means are identical by construction. If they are not, the loop is wrong.
col_means <- colMeans(ranks)
if (diff(range(col_means)) > 1e-9)
  stop("rank columns are not comparable -- transform is not per-sample")
log_msg("  per-patient mean rank: ", sprintf("%.6f", col_means[1]),
        " (identical across patients, as it must be)")

# ---------------------------------------------------------------
# Write. Genes are rows and patients are columns: 517 parquet columns
# instead of ~20,000, which keeps the file readable in one pass. Python
# transposes on load.
# ---------------------------------------------------------------
as_tbl <- function(m) data.table(gene_id = rownames(m), as.data.table(m))

out_counts <- file.path(PROCESSED, paste0(prefix, "_counts.parquet"))
out_rank   <- file.path(PROCESSED, paste0(prefix, "_rank.parquet"))
out_meta   <- file.path(PROCESSED, paste0(prefix, "_gene_meta.parquet"))

# gzip rather than zstd: this arrow build has no zstd codec, and a file the
# collaborator cannot open is worse than one that is 20% larger.
arrow::write_parquet(as_tbl(counts), out_counts, compression = "gzip")
arrow::write_parquet(as_tbl(ranks),  out_rank,   compression = "gzip")
arrow::write_parquet(
  rd[, .(gene_id, gene_name, gene_type,
         chromosome = if ("seqnames" %in% names(rd)) as.character(seqnames) else NA_character_)],
  out_meta)

mb <- function(p) sprintf("%.1f MB", file.size(p) / 1024^2)
log_msg("=== written ===")
log_msg("  ", out_counts, "  (", mb(out_counts), ")")
log_msg("  ", out_rank,   "  (", mb(out_rank),   ")")
log_msg("  ", out_meta,   "  (", nrow(rd), " genes)")
log_msg(nrow(counts), " ", want, " genes x ", ncol(counts), " patients")
log_msg(if (prefix == "rna") "Next: python prep/04_preprocess_mutation.py"
        else "Done.")
