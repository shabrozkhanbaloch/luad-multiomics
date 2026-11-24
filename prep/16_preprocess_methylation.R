# =============================================================
# 16_preprocess_methylation.R -- the DNA methylation block
#
# What it is for
#   Joins the downloaded methylation files of TCGA-LUAD, one per tumour
#   sample, into one table with one column per patient, and removes the
#   probes that cannot be used.
#
#   Words used below:
#   - DNA methylation is a chemical mark on the DNA that can switch a
#     gene off without changing its sequence.
#   - The measurements come from the Illumina 450k array, which reads
#     more than 480,000 fixed positions in the genome. Each position is
#     measured by a probe, named like cg00000029.
#   - A beta value is the result for one probe in one sample: the
#     estimated share of the DNA copies that are methylated there,
#     from 0 (none) to 1 (all).
#   - "Level 3" is the GDC's name for processed, ready-to-use values.
#
# Where it sits
#   Step 9 of 9 of `make prep`, the last one. It needs `make
#   download-methylation` to have been run first.
#
# Reads    data/raw/methylation/GDCdata/   the beta files, one per aliquot
#          data/raw/methylation/HM450.hg38.manifest.tsv.gz   the probe
#              annotation: which chromosome each probe is on. No script of
#              this repository downloads this file; it must be put there.
#          config/config.yaml              (the output folder)
#          It also sends one query to the GDC over the internet, to learn
#          which sample each downloaded file belongs to.
# Writes   data/processed/meth_beta.parquet        probe_id, then one column
#                                                  per patient
#          data/processed/meth_probe_meta.parquet  probe_id, chromosome
#
# How to run (from the project folder)
#   Rscript prep/16_preprocess_methylation.R
#
# WHY METHYLATION IS A BLOCK
# --------------------------
# Methylation is different in kind from the expression block. It measures a
# regulatory state -- whether a locus is switched off -- rather than how
# much transcript there is. It can therefore carry information that
# expression does not.
#
# The published evidence did not promise a gain. Li et al. (2024) ran all
# 31 combinations of five omics types across fourteen cancers and found
# that adding data types "most often resulted in a decline in performance",
# though they note methylation helped in some cancers. Vale-Silva and Rohr
# (2021) put methylation at 0.736 across 33 cancers, below clinical at
# 0.809. In this project, too, methylation did not improve on the clinical
# covariates (README, finding 1).
#
# WHAT HAPPENS HERE, AND WHY EACH STEP CANNOT LEAK
# ------------------------------------------------
# "Leak" means that patients later used to test a model have already
# influenced how its features were prepared (see the header of
# prep/03_preprocess_rna.R). Here, probes are only filtered by annotation
# and by missing values. Nothing is fitted, and nothing depends on the
# outcome or on which patients share a cross-validation fold. The choice of
# the most variable probes, which does depend on the data, is in
# src/luad/transforms.py and runs on the training patients of each fold,
# exactly as it does for expression.
#
#   1. Primary tumour only, one aliquot per patient by the same fixed rule the
#      expression block uses -- prefer vial A, then the lowest plate -- so the
#      choice cannot depend on processing order.
#   2. Sex-chromosome probes dropped. Beta values on chrX and chrY track sex
#      rather than disease, and sex is already a clinical covariate; leaving
#      them in lets the block relearn a variable the clinical model has.
#   3. Probes with any missing value across the cohort dropped. This is the one
#      step that reads all patients at once, and it is safe for the same reason
#      the all-zero gene filter is safe in the expression script: a probe that failed for a patient
#      has failed in every subset containing them, so no fold can learn anything
#      from its removal. Filling in the missing values instead (imputing) would
#      be a fitted step and would have to move inside the fold.
#
#      It costs 146,696 probes of 486,427, which looks severe until the missing
#      values are counted per array. Every one of the 473 arrays masks between
#      65,896 and 102,983 probes -- a median of 68,986, about 14% -- because
#      GDC's sesame pipeline masks per-sample quality failures. There is no
#      subset of bad arrays to drop instead: the minimum is 13.5% and the
#      maximum 21%. The union across 458 patients is 33%, so the masked sets
#      overlap heavily and 327,214 probes survive, which is close to the best
#      any complete-case rule could do. The alternative, imputing, would buy
#      those probes at the price of a fitted step inside every fold, for a
#      feature space the variance filter cuts to at most 2,000 anyway.
#
# Beta values are proportions in [0, 1] and are already comparable across
# samples, so no transform is applied. The expression block needs its
# within-sample rank because counts are not comparable across platforms; that
# problem does not arise here, and inventing a transform to match would add a
# step without a reason.
# =============================================================

# yaml reads the configuration, arrow writes parquet files, data.table reads
# and handles tables, TCGAbiolinks sends the query to the GDC.
suppressPackageStartupMessages({
  library(yaml); library(arrow); library(data.table); library(TCGAbiolinks)
})

# The output folder comes from the configuration. The two input locations
# are fixed here: the folder that `make download-methylation` fills, and
# the probe annotation file.
cfg       <- yaml::read_yaml("config/config.yaml")
PROCESSED <- cfg$paths$processed
GDCDIR    <- "data/raw/methylation/GDCdata"
MANIFEST  <- "data/raw/methylation/HM450.hg38.manifest.tsv.gz"

# Print a message with the time of day in front of it.
log_msg <- function(...) cat(format(Sys.time(), "[%H:%M:%S] "), ..., "\n", sep = "")

# ---------------------------------------------------------------
# Find the beta files and work out which sample each one belongs to.
#
# The files are read directly. TCGAbiolinks' GDCprepare() is not used: it
# routes methylation through the package sesame, which is not installed and
# would add a large dependency for work these files do not need. Each file
# is two columns -- probe id and beta -- with no header, and every one
# carries the same 486,427 probes in the same order. That is checked below
# for every file, not assumed.
#
# Both inputs must be present. The second message mentions a Makefile
# section "methylation-block" that this repository does not have; the file
# it asks for is the probe annotation described in the header.
# ---------------------------------------------------------------
if (!dir.exists(GDCDIR)) stop(GDCDIR, " not found -- run `make download-methylation`")
if (!file.exists(MANIFEST)) stop(MANIFEST, " not found -- run `make download-methylation`")

# All beta files, in whatever sub-folders the download put them.
log_msg("=== locating files ===")
files <- list.files(GDCDIR, pattern = "level3betas\\.txt$",
                    recursive = TRUE, full.names = TRUE)
log_msg("  beta files on disk: ", length(files))

# A downloaded file is named after a file identifier, not after the sample.
# The same query that `make download-methylation` used is sent again (this
# needs the internet). Its result lists, for every file name, the barcode
# of the aliquot it was measured on (column `cases`). `map` turns a file
# name into that barcode. The script stops if a file cannot be matched.
# The query asks for primary tumours only, so no other sample type arrives.
q <- GDCquery(project = "TCGA-LUAD", data.category = "DNA Methylation",
              platform = "Illumina Human Methylation 450",
              data.type = "Methylation Beta Value", sample.type = "Primary Tumor")
res <- as.data.table(getResults(q))[, .(file_name, cases)]
res[, base := basename(file_name)]
map <- res$cases; names(map) <- res$base
barcode <- unname(map[basename(files)])
if (anyNA(barcode)) stop(sum(is.na(barcode)), " files could not be matched to a barcode")

# ---------------------------------------------------------------
# One aliquot per patient -- the same rule prep/03_preprocess_rna.R uses.
#
# The barcode is cut into its parts: characters 1-12 are the patient,
# character 16 the vial, characters 22-25 the plate (explained in
# prep/03_preprocess_rna.R). A patient measured more than once keeps one
# aliquot: the first after sorting by vial letter and then plate number.
# Keeping both would count the patient twice.
# ---------------------------------------------------------------
log_msg("=== sample selection ===")
dt <- data.table(path = files, barcode = barcode,
                 patient = substr(barcode, 1, 12),
                 vial = substr(barcode, 16, 16),
                 plate = substr(barcode, 22, 25))
setorder(dt, patient, vial, plate)
keep <- dt[, .SD[1], by = patient]
log_msg("  aliquots: ", nrow(dt), " -> one per patient: ", nrow(keep))

# ---------------------------------------------------------------
# Assemble one matrix: probes in rows, patients in columns.
#
# The first file sets the list and order of the probes. Every later file
# must have exactly the same probes in the same order, because its values
# are copied in by position; the script stops if one does not. Progress is
# printed every 100 files.
# ---------------------------------------------------------------
log_msg("=== reading ", nrow(keep), " files ===")
first <- fread(keep$path[1], header = FALSE, col.names = c("probe", "beta"))
probes <- first$probe
beta <- matrix(NA_real_, nrow = length(probes), ncol = nrow(keep),
               dimnames = list(probes, keep$patient))
beta[, 1] <- first$beta
for (i in seq_len(nrow(keep))[-1]) {
  d <- fread(keep$path[i], header = FALSE, col.names = c("probe", "beta"))
  if (!identical(d$probe, probes))
    stop("probe order differs in ", basename(keep$path[i]),
         " -- the matrix cannot be assembled positionally")
  beta[, i] <- d$beta
  if (i %% 100 == 0) { log_msg("  ", i, " / ", nrow(keep)); flush.console() }
}
log_msg("  assembled: ", nrow(beta), " probes x ", ncol(beta), " patients")

# ---------------------------------------------------------------
# Probe filtering -- first by annotation, then by missing values.
#
# The annotation file gives the chromosome of each probe. Three kinds of
# probe are removed, and each count is logged:
#   - probes with no position in the annotation;
#   - probes on chrX or chrY (reason 2 in the header);
#   - probes with a missing value in at least one patient (reason 3).
# `chr` is kept in step with `beta`, so that it still lists the chromosome
# of every remaining probe.
# ---------------------------------------------------------------
log_msg("=== probe filtering ===")
man <- fread(MANIFEST, select = c("Probe_ID", "CpG_chrm"))
setnames(man, c("probe_id", "chromosome"))
chr <- man[match(rownames(beta), man$probe_id)]$chromosome

unmapped <- is.na(chr)
log_msg("  probes with no mapped position dropped: ", sum(unmapped))
sex_probe <- chr %in% c("chrX", "chrY")
log_msg("  sex-chromosome probes dropped: ", sum(sex_probe, na.rm = TRUE),
        "   (sex is already a clinical covariate; leaving them in lets the",
        " block relearn it)")
drop <- unmapped | sex_probe
beta <- beta[!drop, , drop = FALSE]
chr  <- chr[!drop]

na_per_probe <- rowSums(is.na(beta))
log_msg("  probes with any missing value dropped: ", sum(na_per_probe > 0),
        "   (kept ", sum(na_per_probe == 0), ")")
beta <- beta[na_per_probe == 0, , drop = FALSE]
chr  <- chr[na_per_probe == 0]

# A last check: a beta value is a proportion, so all values must lie
# between 0 and 1. Anything else means the wrong kind of file was read.
r <- range(beta)
log_msg("  beta range: [", sprintf("%.4f", r[1]), ", ", sprintf("%.4f", r[2]), "]")
if (r[1] < 0 || r[2] > 1) stop("beta values outside [0, 1] -- wrong assay?")

# ---------------------------------------------------------------
# Write the two files. as_tbl() puts the probe identifiers into a first
# column called probe_id, because a parquet file has no row names. The
# second file lists the chromosome of each probe that was kept.
# ---------------------------------------------------------------
as_tbl <- function(m) data.table(probe_id = rownames(m), as.data.table(m))
out_beta <- file.path(PROCESSED, "meth_beta.parquet")
out_meta <- file.path(PROCESSED, "meth_probe_meta.parquet")
arrow::write_parquet(as_tbl(beta), out_beta, compression = "gzip")
arrow::write_parquet(data.table(probe_id = rownames(beta), chromosome = chr), out_meta)

# Report what was written. This is the last prep step; the analysis
# (scripts/01_cv.py, or `make all`) can run next.
mb <- function(p) sprintf("%.1f MB", file.size(p) / 1024^2)
log_msg("=== written ===")
log_msg("  ", out_beta, "  (", mb(out_beta), ")")
log_msg("  ", out_meta, "  (", nrow(beta), " probes)")
log_msg(nrow(beta), " probes x ", ncol(beta), " patients")
log_msg("The data are prepared. Next: make all")
