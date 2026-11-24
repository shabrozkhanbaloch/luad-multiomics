# =============================================================
# 16_preprocess_methylation.R -- the DNA methylation block
#
#   Rscript prep/16_preprocess_methylation.R
#
# Writes:
#   data/processed/meth_beta.parquet       probe_id, then one column per patient
#   data/processed/meth_probe_meta.parquet probe_id, chromosome, gene, island
#
# WHY THIS BLOCK, AND WHY IT IS THE LAST ONE WORTH BUILDING
# ---------------------------------------------------------
# A2 Section 5.5 states what would overturn RQ1: "an omics representation that
# captures something OTHER than the large-scale axes bulk expression is
# dominated by". Neither alternative already tested meets that. lncRNA is a
# different slice of the same assay; Hallmark scores are the same axes
# reorganised. Both came back null, as Section 5.5 predicted they would.
#
# Methylation is the first block that is different in kind. It measures a
# regulatory state -- whether a locus is switched off -- rather than transcript
# abundance. It is the one remaining candidate that the dissertation's own
# criterion points at, which is why it is being built and the others are not.
#
# The prior is still against it. Li et al. (2024) ran all 31 combinations of
# five omics types across fourteen cancers and found that adding data types
# "most often resulted in a decline in performance", though they note
# methylation helped in some cancers. Vale-Silva and Rohr (2021) put methylation
# at 0.736 across 33 cancers, below clinical at 0.809. This project's own three
# blocks (0.554) already score below its two (0.565). A null here is the
# expected result and closes an eighth objection; a positive would be the first
# representation to meet Section 5.5's bar.
#
# WHAT HAPPENS HERE, AND WHY EACH STEP CANNOT LEAK
# ------------------------------------------------
# Only annotation-driven filtering and a per-probe read. Nothing is fitted, and
# nothing depends on the outcome or on which patients share a fold. The
# variance prefilter and every other data-dependent step live in
# utils/foldwise.py and run inside the training partition, exactly as they do
# for expression.
#
#   1. Primary tumour only, one aliquot per patient by the same fixed rule the
#      expression block uses -- prefer vial A, then the lowest plate -- so the
#      choice cannot depend on processing order.
#   2. Sex-chromosome probes dropped. Beta values on chrX and chrY track sex
#      rather than disease, and sex is already a clinical covariate; leaving
#      them in lets the block relearn a variable the clinical model has.
#   3. Probes with any missing value across the cohort dropped. This is the one
#      step that reads all patients at once, and it is safe for the same reason
#      the all-zero gene filter is safe in 03: a probe that failed for a patient
#      has failed in every subset containing them, so no fold can learn anything
#      from its removal. Imputing instead would be a fitted step and would have
#      to move inside the fold.
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
#      feature space the variance prefilter cuts to at most 2,000 anyway.
#
# Beta values are proportions in [0, 1] and are already comparable across
# samples, so no transform is applied. The expression block needs its
# within-sample rank because counts are not comparable across platforms; that
# problem does not arise here, and inventing a transform to match would add a
# step without a reason.
# =============================================================

suppressPackageStartupMessages({
  library(yaml); library(arrow); library(data.table); library(TCGAbiolinks)
})

cfg       <- yaml::read_yaml("config/config.yaml")
PROCESSED <- cfg$paths$processed
GDCDIR    <- "data/raw/methylation/GDCdata"
MANIFEST  <- "data/raw/methylation/HM450.hg38.manifest.tsv.gz"

log_msg <- function(...) cat(format(Sys.time(), "[%H:%M:%S] "), ..., "\n", sep = "")

# ---------------------------------------------------------------
# Read the level-3 beta files directly.
#
# GDCprepare() is not used: it routes methylation through sesame, which is not
# installed and would add a large dependency for work these files do not need.
# Each file is two columns -- probe id and beta -- with no header, and every one
# carries the same 486,427 probes in the same order, which is asserted below
# rather than assumed.
# ---------------------------------------------------------------
if (!dir.exists(GDCDIR)) stop(GDCDIR, " not found -- run `make download-methylation`")
if (!file.exists(MANIFEST)) stop(MANIFEST, " not found -- see Makefile methylation-block")

log_msg("=== locating files ===")
files <- list.files(GDCDIR, pattern = "level3betas\\.txt$",
                    recursive = TRUE, full.names = TRUE)
log_msg("  beta files on disk: ", length(files))

q <- GDCquery(project = "TCGA-LUAD", data.category = "DNA Methylation",
              platform = "Illumina Human Methylation 450",
              data.type = "Methylation Beta Value", sample.type = "Primary Tumor")
res <- as.data.table(getResults(q))[, .(file_name, cases)]
res[, base := basename(file_name)]
map <- res$cases; names(map) <- res$base
barcode <- unname(map[basename(files)])
if (anyNA(barcode)) stop(sum(is.na(barcode)), " files could not be matched to a barcode")

# ---------------------------------------------------------------
# One aliquot per patient -- the same vial-then-plate rule 03 uses
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
# Assemble
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
# Probe filtering -- annotation, then structural missingness
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

r <- range(beta)
log_msg("  beta range: [", sprintf("%.4f", r[1]), ", ", sprintf("%.4f", r[2]), "]")
if (r[1] < 0 || r[2] > 1) stop("beta values outside [0, 1] -- wrong assay?")

# ---------------------------------------------------------------
# Write
# ---------------------------------------------------------------
as_tbl <- function(m) data.table(probe_id = rownames(m), as.data.table(m))
out_beta <- file.path(PROCESSED, "meth_beta.parquet")
out_meta <- file.path(PROCESSED, "meth_probe_meta.parquet")
arrow::write_parquet(as_tbl(beta), out_beta, compression = "gzip")
arrow::write_parquet(data.table(probe_id = rownames(beta), chromosome = chr), out_meta)

mb <- function(p) sprintf("%.1f MB", file.size(p) / 1024^2)
log_msg("=== written ===")
log_msg("  ", out_beta, "  (", mb(out_beta), ")")
log_msg("  ", out_meta, "  (", nrow(beta), " probes)")
log_msg(nrow(beta), " probes x ", ncol(beta), " patients")
log_msg("Next: python scripts/01_cv.py")
