#!/usr/bin/env Rscript
# =============================================================
# 03_preprocess_rna.R -- the gene expression (RNA-seq) block
#
# What it is for
#   Turns the downloaded TCGA-LUAD RNA-seq object into the three expression
#   files the analysis reads: one tumour sample per patient, protein-coding
#   genes only, as raw counts and as within-sample ranks.
#
# Where it sits
#   Step 4 of 9 of `make prep`, after `make download`.
#
# Reads    data/raw/tcga_luad_rnaseq_se.rds   (written by 01_download.R)
#          config/config.yaml                 (sample type, gene biotype)
# Writes   data/processed/rna_counts.parquet     raw STAR counts
#          data/processed/rna_rank.parquet       within-sample ranks in (0, 1]
#          data/processed/rna_gene_meta.parquet  gene id, symbol, biotype,
#                                                chromosome
#          In the first two files a row is a gene and a column is a patient,
#          named by the 12-character patient barcode.
#
# How to run (from the project folder)
#   Rscript prep/03_preprocess_rna.R
#
# WHY BOTH COUNTS AND RANKS ARE KEPT
# ----------------------------------
# The ranks are what the models are fitted on. The counts are still needed
# later, in two places: to decide inside each cross-validation fold which
# genes are expressed at all (src/luad/transforms.py), and to recompute
# the ranks over the genes shared with a GEO cohort (src/luad/external.py).
#
# WHAT THIS SCRIPT DELIBERATELY DOES NOT DO
# -----------------------------------------
# Some usual preprocessing steps learn something from the whole group of
# patients they are given: dropping weakly expressed genes, keeping the
# most variable genes, variance-stabilising, scaling each gene to mean 0.
# The analysis tests its models on patients that were held out
# (cross-validation; each split of the patients is called a fold). If such
# a step were run here, once, over all 517 patients, the held-out patients
# would already have shaped the features of the model that is then tested
# on them. That is called leakage, and it makes results look better than
# they are. The shuffled-outcome check (`make permute`) cannot detect this
# kind of leak, because these steps never look at the outcome.
# The steps of this kind that the project uses, the expression filter and
# the choice of the most variable genes, are therefore in
# src/luad/transforms.py and are fitted on the training patients of each
# fold only.
#
# Only two things happen here, and both are safe by construction:
#
#   1. Selection by annotation (protein-coding genes, primary tumour, one
#      aliquot per patient). It depends on the gene annotation and on the
#      sample barcode, never on the expression values.
#   2. The within-sample rank transform. Each patient's ranks are computed
#      across that patient's own genes, and do not change with which other
#      patients are in the fold. That property is why the rank is the
#      representation the models use.
#
# The single exception is dropping genes with zero counts in every one of
# the 517 patients. Such a gene is necessarily all-zero in any subset, so
# no fold can learn anything from its removal.
# =============================================================

# yaml reads the configuration, arrow writes parquet files (a compact table
# format that both R and Python read), data.table handles tables, and
# SummarizedExperiment is the object the download was saved as.
suppressPackageStartupMessages({
  library(yaml); library(arrow); library(data.table)
  library(SummarizedExperiment)
})

# Folders from the configuration. The seed is set as a precaution; nothing
# in this script is random.
cfg       <- yaml::read_yaml("config/config.yaml")
RAW       <- cfg$paths$raw
PROCESSED <- cfg$paths$processed
set.seed(cfg$project$seed)
dir.create(PROCESSED, recursive = TRUE, showWarnings = FALSE)

# Print a message with the time of day in front of it.
log_msg <- function(...) cat(format(Sys.time(), "[%H:%M:%S] "), ..., "\n", sep = "")

# ---------------------------------------------------------------
# Reading a TCGA barcode.
#
# Every TCGA sample has a barcode, an identifier such as
#   TCGA-XX-XXXX-01A-11R-XXXX-07
# whose parts have fixed positions:
#   characters  1-12  the patient
#   characters 14-15  the sample type: 01 is a primary solid tumour,
#                     11 is normal tissue
#   character  16     the vial: a letter, A for the first, when a sample
#                     was divided into several vials
#   characters 22-25  the plate on which the material was processed
# An aliquot is the portion of a sample's material that was actually
# sequenced. The full barcode names one aliquot, and one patient can have
# more than one. The four helpers below cut these parts out of a barcode.
# ---------------------------------------------------------------
patient_of     <- function(bc) substr(bc, 1, 12)
sample_type_of <- function(bc) substr(bc, 14, 15)
vial_of        <- function(bc) substr(bc, 16, 16)
plate_of       <- function(bc) substr(bc, 22, 25)

# The input. If it is missing, the download step has not been run.
f <- file.path(RAW, "tcga_luad_rnaseq_se.rds")
if (!file.exists(f)) stop("missing ", f, "\n  run `make download` first")

log_msg("=== loading SummarizedExperiment ===")
se <- readRDS(f)
log_msg("  ", nrow(se), " genes x ", ncol(se), " aliquots")

# The object holds several tables ("assays") for the same genes and samples.
# `unstranded` is the one with the raw counts: for each gene, the number of
# sequencing reads assigned to it. Everything below assumes whole numbers,
# so the script stops if that table is not there.
if (!("unstranded" %in% assayNames(se)))
  stop("no `unstranded` assay; present: ", paste(assayNames(se), collapse = ", "))

# ---------------------------------------------------------------
# Sample selection.
#
# Step 1: keep primary solid tumours (sample type 01, set in the
# configuration). Normal tissue and other sample types are dropped.
#
# Step 2: keep one aliquot per patient. Some patients have more than one
# aliquot sequenced. Keeping all of them would count those patients twice,
# and one patient could then be in the training set and the test set at
# the same time. The one to keep is chosen by a fixed rule that never
# looks at the data: sort by vial letter, then by plate number, and take
# the first. A fixed rule gives the same choice on every run.
#
# The same two steps are in prep/utils/export_barcodes.R and must stay
# identical there, or the patient counts in data/interim/consort_counts.csv
# and the expression matrix would describe different groups of patients.
# ---------------------------------------------------------------
log_msg("=== sample selection ===")
bc   <- colnames(se)
keep <- sample_type_of(bc) %in% cfg$data$tcga$sample_types
log_msg("  primary solid tumour (type 01): ", sum(keep),
        "  (dropped ", sum(!keep), ")")
se <- se[, keep]

# `dt` has one row per aliquot. `n_dup` counts the extra aliquots. After the
# sort, `.SD[1]` takes the first row of each patient. The columns of `se`
# are then renamed from the aliquot barcode to the patient barcode.
dt <- data.table(barcode = colnames(se), patient = patient_of(colnames(se)),
                 vial = vial_of(colnames(se)), plate = plate_of(colnames(se)))
n_dup <- sum(duplicated(dt$patient))
setorder(dt, patient, vial, plate)
dt <- dt[, .SD[1], by = patient]
se <- se[, dt$barcode]
colnames(se) <- dt$patient
log_msg("  one aliquot per patient: ", ncol(se), "  (resolved ", n_dup, " duplicates)")

# A safety check: every patient must now appear exactly once.
if (anyDuplicated(colnames(se))) stop("duplicate patient barcodes survived deduplication")

# ---------------------------------------------------------------
# Gene selection -- by annotation only.
#
# rowData() is the description of the genes: identifier, symbol and
# biotype (protein_coding, lncRNA, ...). Only the biotype named in the
# configuration is kept, which is protein_coding. The choice uses the
# annotation alone and no expression value, so it cannot leak.
# ---------------------------------------------------------------
log_msg("=== gene selection ===")
rd <- as.data.table(rowData(se))
rd[, gene_id := rownames(se)]

# The biotype and the prefix of the output files can be changed on the
# command line, so that a second block can be built from the SAME download
# without touching the protein-coding files. The defaults give the three
# files the analysis reads:
#
#   Rscript prep/03_preprocess_rna.R                        -> rna_*.parquet
#   Rscript prep/03_preprocess_rna.R --biotype lncRNA --prefix lnc
#                                                          -> lnc_*.parquet
#
# The option comes from the earlier project, which tested long non-coding
# RNA (lncRNA) as a block: Ellen et al. (2023) report a multi-omics gain in
# LUAD from lncRNA, and the STAR-counts object already carries 16,901 of
# them. The analysis in this repository does not use an lncRNA block.
#
# argval() returns the value that follows a flag on the command line, or
# the default if the flag is not given.
args   <- commandArgs(trailingOnly = TRUE)
argval <- function(flag, default) {
  i <- match(flag, args)
  if (is.na(i) || i == length(args)) default else args[[i + 1L]]
}
want   <- argval("--biotype", cfg$preprocessing$expression$gene_biotype)
prefix <- argval("--prefix",  "rna")

# Stop if the biotype asked for does not exist in the annotation.
if (!(want %in% rd$gene_type))
  stop("biotype '", want, "' absent from rowData$gene_type; present: ",
       paste(head(sort(unique(rd$gene_type)), 20), collapse = ", "))

# `is_pc` is TRUE for the genes of the wanted biotype (protein-coding by
# default). Both the data and the gene description are cut to those genes.
is_pc <- rd$gene_type == want
log_msg("  ", want, ": ", sum(is_pc), "  (dropped ", sum(!is_pc), " other biotypes)")
se <- se[is_pc, ]
rd <- rd[is_pc]

# Take the table of raw counts out of the object and store it as integers.
counts <- assay(se, "unstranded")
storage.mode(counts) <- "integer"

# Drop genes that have a count of zero in every patient: they carry no
# information. A gene that is all-zero across the full cohort is all-zero
# in every possible training subset too, so this cannot leak.
nz <- rowSums(counts) > 0
log_msg("  non-zero in >= 1 patient: ", sum(nz), "  (dropped ", sum(!nz),
        " all-zero genes)")
counts <- counts[nz, , drop = FALSE]
rd     <- rd[nz]

# The counts and the gene description must still list the same genes in
# the same order.
stopifnot(identical(rownames(counts), rd$gene_id))

# ---------------------------------------------------------------
# Within-sample rank transform -- the representation the models use.
#
# For each patient separately, the genes are ranked from the lowest count
# to the highest, and the rank is divided by the number of genes. Every
# value therefore lies in (0, 1]: about 0.5 means "in the middle of this
# patient's genes", 1 means "this patient's most expressed gene".
#
# Why ranks: the total number of reads per sample, the measuring platform
# and the shape of each cohort's distribution all drop out. That is what
# makes a model fitted on TCGA RNA-seq usable on GEO microarrays at all.
#
# Tied values get the average of their ranks (ties.method = "average").
# The many genes with a count of zero in a patient therefore share one low
# rank. That is intended: it records "not detected" as a single level
# instead of inventing an order among genes that were not seen.
# ---------------------------------------------------------------
log_msg("=== within-sample rank transform ===")
n_genes <- nrow(counts)
ranks <- matrix(0, nrow = n_genes, ncol = ncol(counts),
                dimnames = dimnames(counts))
for (j in seq_len(ncol(counts))) {
  ranks[, j] <- rank(counts[, j], ties.method = "average") / n_genes
}
storage.mode(ranks) <- "double"

# Check the range, and print it.
stopifnot(all(ranks > 0), all(ranks <= 1))
log_msg("  range: [", sprintf("%.4f", min(ranks)), ", ",
        sprintf("%.4f", max(ranks)), "]  over ", n_genes, " genes")
# A second check. Ranking n genes always uses up the same total of ranks,
# whatever the ties, so every patient's mean rank must be the same number.
# If the means differ, the transform was not done patient by patient.
col_means <- colMeans(ranks)
if (diff(range(col_means)) > 1e-9)
  stop("rank columns are not comparable -- transform is not per-sample")
log_msg("  per-patient mean rank: ", sprintf("%.6f", col_means[1]),
        " (identical across patients, as it must be)")

# ---------------------------------------------------------------
# Write the three files.
#
# Genes are rows and patients are columns: 517 parquet columns instead of
# ~20,000, which keeps the file readable in one pass. Python transposes on
# load. as_tbl() puts the gene identifiers into a first column called
# gene_id, because a parquet file has no row names.
# ---------------------------------------------------------------
as_tbl <- function(m) data.table(gene_id = rownames(m), as.data.table(m))

out_counts <- file.path(PROCESSED, paste0(prefix, "_counts.parquet"))
out_rank   <- file.path(PROCESSED, paste0(prefix, "_rank.parquet"))
out_meta   <- file.path(PROCESSED, paste0(prefix, "_gene_meta.parquet"))

# Compression is gzip rather than zstd: the arrow installation this was
# written with had no zstd codec, and a file that cannot be opened
# everywhere is worse than one that is 20% larger.
# The third file is the gene description; its chromosome column is empty
# if the annotation has no `seqnames` column.
arrow::write_parquet(as_tbl(counts), out_counts, compression = "gzip")
arrow::write_parquet(as_tbl(ranks),  out_rank,   compression = "gzip")
arrow::write_parquet(
  rd[, .(gene_id, gene_name, gene_type,
         chromosome = if ("seqnames" %in% names(rd)) as.character(seqnames) else NA_character_)],
  out_meta)

# Report what was written and how large it is. With the default prefix the
# last line names the script that follows in the numbering.
mb <- function(p) sprintf("%.1f MB", file.size(p) / 1024^2)
log_msg("=== written ===")
log_msg("  ", out_counts, "  (", mb(out_counts), ")")
log_msg("  ", out_rank,   "  (", mb(out_rank),   ")")
log_msg("  ", out_meta,   "  (", nrow(rd), " genes)")
log_msg(nrow(counts), " ", want, " genes x ", ncol(counts), " patients")
log_msg(if (prefix == "rna") "Next: python prep/04_preprocess_mutation.py"
        else "Done.")
