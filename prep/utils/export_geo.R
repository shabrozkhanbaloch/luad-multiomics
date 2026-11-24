#!/usr/bin/env Rscript
# =============================================================
# export_geo.R -- the three external GEO cohorts as plain tables
#
# What it is for
#   Turns each filtered GEO cohort into the two tables that the external
#   validation reads (scripts/03_external.py, src/luad/external.py):
#
#   data/processed/geo_<ID>_expr.parquet      one row per gene (column
#                                             gene_name), one column per sample
#   data/processed/geo_<ID>_clinical.parquet  one row per sample: outcomes
#                                             and four covariates
#
# Where it sits
#   Step 8 of 9 of `make prep`. It needs prep/01b_filter_geo.R (step 1).
#
# Reads    data/interim/<ID>_tumour_eset.rds  (written by 01b_filter_geo.R)
#          config/config.yaml                 (the list of GEO series)
# Writes   the two files above for each of GSE72094, GSE31210 and GSE68465.
#
# How to run (from the project folder)
#   Rscript prep/utils/export_geo.R
#
# Three cohorts, three microarray platforms, three ways of writing the
# clinical information. Every decision below is about bringing them
# together WITHOUT quietly inventing agreement that is not there.
#
# Probe -> gene symbol
#   A microarray measures probes, short sequences that each detect one
#   transcript; a gene can have several probes. The models work with genes.
#   A symbol measured by several probes keeps the probe with the highest
#   mean intensity, which is the standard collapse rule and favours the
#   probe with the best dynamic range. Probes annotated to several symbols
#   ("DDR1 /// MIR4640") are DROPPED rather than assigned to the first:
#   an ambiguous probe is not evidence about either gene.
#
# Expression scale
#   GSE31210 and GSE68465 arrive on a linear intensity scale, GSE72094 on
#   log2. Nothing is done about that here ON PURPOSE. The models use the
#   within-sample RANK of each gene, and a rank does not change under any
#   transform that keeps the order of a sample's values, so log and linear
#   give the identical representation. That is the reason the project uses
#   ranks. The ranks are not computed here: the analysis computes them
#   itself, over the genes a cohort shares with TCGA (src/luad/external.py).
#
# Endpoints
#   All three cohorts record overall survival (exported as os_time and
#   os_event). GSE31210 and GSE68465 also record relapse or progression,
#   exported as rfs_time and rfs_event (relapse-free survival), which is
#   the closest match to TCGA's progression-free interval. GSE72094 does
#   not record it, so its rfs columns are empty. The config (external:
#   endpoints) pairs TCGA's PFI with rfs and TCGA's OS with os.
#   Times are exported in months. An event column is 1 if the event was
#   seen and 0 if the patient was censored, that is, last seen without it.
# =============================================================
# Biobase provides the ExpressionSet object, arrow writes parquet files,
# yaml reads the configuration.
suppressPackageStartupMessages({
  library(Biobase); library(arrow); library(yaml)
})

# `out` is the folder for prepared data (data/processed).
cfg <- yaml::read_yaml("config/config.yaml")
out <- cfg$paths$processed
dir.create(out, recursive = TRUE, showWarnings = FALSE)

# Small helpers.
# %||% gives `a` unless it is empty, and `b` otherwise. (It is defined here
# but not used further down.)
`%||%` <- function(a, b) if (is.null(a) || length(a) == 0) b else a

# num() turns text into a number. Text that is not a number becomes NA,
# without a warning.
num <- function(x) suppressWarnings(as.numeric(trimws(as.character(x))))

# GEO writes "no value" as "--", "NA", "Unknown", "Missing" and "".
# blank() turns all of these into NA.
blank <- function(x) {
  x <- trimws(as.character(x))
  x[x %in% c("--", "NA", "", "Unknown", "unknown", "Missing", "missing")] <- NA
  x
}

# The probe description of each platform names its gene-symbol column
# differently. symbol_column() finds it, or stops.
symbol_column <- function(fd) {
  hit <- intersect(c("Gene Symbol", "GeneSymbol", "SYMBOL", "gene_symbol"),
                   colnames(fd))
  if (!length(hit)) stop("no gene-symbol column in featureData")
  hit[1]
}

# collapse_probes() goes from one row per probe to one row per gene symbol,
# by the rule described in the header. `fd` is the probe description, `sym`
# the gene symbol of each probe, `x` the expression values.
collapse_probes <- function(e) {
  fd <- fData(e)
  sym <- blank(fd[[symbol_column(fd)]])
  x   <- exprs(e)

  # Drop probes without a symbol, and probes that list several symbols
  # (written with "///" between them): such a probe says nothing certain
  # about any one gene. The counts are printed.
  ambiguous <- grepl("///", sym, fixed = TRUE)
  drop <- is.na(sym) | sym == "" | ambiguous
  cat(sprintf("    probes %6d  ->  dropped %5d (%d unannotated, %d multi-symbol)\n",
              nrow(x), sum(drop), sum(is.na(sym) | sym == ""), sum(ambiguous)))
  x <- x[!drop, , drop = FALSE]; sym <- sym[!drop]

  # Highest mean intensity wins the symbol: sort the probes from the
  # highest mean to the lowest, then keep the first probe of each symbol.
  # The result is returned sorted by gene symbol.
  ord  <- order(rowMeans(x, na.rm = TRUE), decreasing = TRUE)
  x    <- x[ord, , drop = FALSE]; sym <- sym[ord]
  keep <- !duplicated(sym)
  x    <- x[keep, , drop = FALSE]
  rownames(x) <- sym[keep]
  cat(sprintf("    genes  %6d after collapsing to symbols\n", nrow(x)))
  x[order(rownames(x)), , drop = FALSE]
}

# ---- per-cohort outcome and covariate extraction ------------------------
# Deliberately one function per cohort. One general function over three
# incompatible ways of writing the data would be shorter, and would hide
# exactly the decisions a reader needs to check.
#
# Each function takes the sample description (`pd`) and returns the same
# nine columns: sample_id, os_time, os_event, rfs_time, rfs_event, age,
# sex_male, stage_ordinal, ever_smoker. Times are in months; a time given
# in days is divided by 30.4375, the mean length of a month (365.25 / 12).
# Stage is a number from 1 to 4 where the cohort allows it.

clin_GSE31210 <- function(pd) {
  # GSE31210. Both endpoints are recorded in days, each with a status field
  # of two values ("dead" or not, "relapsed" or not).
  data.frame(
    sample_id  = rownames(pd),
    os_time    = num(pd[["days before death/censor:ch1"]]) / 30.4375,
    os_event   = as.integer(blank(pd[["death:ch1"]]) == "dead"),
    rfs_time   = num(pd[["days before relapse/censor:ch1"]]) / 30.4375,
    rfs_event  = as.integer(blank(pd[["relapse:ch1"]]) == "relapsed"),
    age        = num(pd[["age (years):ch1"]]),
    sex_male   = as.integer(blank(pd[["gender:ch1"]]) == "male"),
    # This cohort is stage I-II by design; there is no III or IV in it.
    stage_ordinal = ifelse(grepl("^I[AB]?$", blank(pd[["pathological stage:ch1"]])), 1,
                    ifelse(grepl("^II",      blank(pd[["pathological stage:ch1"]])), 2, NA)),
    ever_smoker = as.integer(blank(pd[["smoking status:ch1"]]) == "Ever-smoker"),
    stringsAsFactors = FALSE)
}

clin_GSE68465 <- function(pd) {
  # GSE68465. Times are already in months. `prog` says whether the patient
  # progressed or relapsed ("Yes"), or is missing.
  prog <- blank(pd[["first_progression_or_relapse:ch1"]])
  d <- data.frame(
    sample_id = rownames(pd),
    os_time   = num(pd[["months_to_last_contact_or_death:ch1"]]),
    os_event  = as.integer(blank(pd[["vital_status:ch1"]]) == "Dead"),
    # The time to progression is recorded only for the patients who
    # progressed. For the others the time of last contact is used, and they
    # count as censored. The rfs columns are therefore built here, not read
    # from one field. If `prog` is missing, the event is missing too.
    rfs_time  = ifelse(!is.na(prog) & prog == "Yes",
                       num(pd[["months_to_first_progression:ch1"]]),
                       num(pd[["months_to_last_contact_or_death:ch1"]])),
    rfs_event = ifelse(is.na(prog), NA_integer_, as.integer(prog == "Yes")),
    age       = num(pd[["age:ch1"]]),
    sex_male  = as.integer(blank(pd[["Sex:ch1"]]) == "Male"),
    stringsAsFactors = FALSE)
  # Stage is recorded as pNxpTy (lymph-node status N and tumour size T), not
  # as an AJCC stage group. Mapping it to one properly needs M status
  # (distant spread), which this cohort does not report. So a coarse scale
  # from node status and tumour size is used:
  #   N0 and T1 or lower -> 1;  N0 with a larger T -> 2;  N1 -> 2;
  #   N2 or higher -> 3;  NA if N or T cannot be read.
  # It is an approximation of the stage, not the stage itself.
  st <- blank(pd[["disease_stage:ch1"]])
  n  <- suppressWarnings(as.integer(sub(".*pN([0-9]).*", "\\1", st)))
  t_ <- suppressWarnings(as.integer(sub(".*pT([0-9]).*", "\\1", st)))
  d$stage_ordinal <- ifelse(is.na(n) | is.na(t_), NA,
                     ifelse(n == 0 & t_ <= 1, 1,
                     ifelse(n == 0,           2,
                     ifelse(n == 1,           2, 3))))
  # Ever-smoker: anything recorded other than "Never smoked".
  sm <- blank(pd[["smoking_history:ch1"]])
  d$ever_smoker <- ifelse(is.na(sm), NA_integer_,
                          as.integer(sm != "Never smoked"))
  d
}

clin_GSE72094 <- function(pd) {
  # GSE72094. Survival time is in days. Relapse is not recorded. The stage
  # number is read from the first character of the stage text, which
  # assumes that the text begins with a digit; anything else becomes NA.
  st <- blank(pd[["Stage:ch1"]])
  sm <- blank(pd[["smoking_status:ch1"]])
  data.frame(
    sample_id = rownames(pd),
    os_time   = num(pd[["survival_time_in_days:ch1"]]) / 30.4375,
    os_event  = ifelse(is.na(blank(pd[["vital_status:ch1"]])), NA_integer_,
                       as.integer(blank(pd[["vital_status:ch1"]]) == "Dead")),
    rfs_time  = NA_real_,        # not recorded in this series
    rfs_event = NA_integer_,
    age       = num(pd[["age_at_diagnosis:ch1"]]),
    sex_male  = as.integer(blank(pd[["gender:ch1"]]) == "M"),
    stage_ordinal = suppressWarnings(as.integer(substr(st, 1, 1))),
    ever_smoker = ifelse(is.na(sm), NA_integer_, as.integer(sm == "Ever")),
    stringsAsFactors = FALSE)
}

# Which function belongs to which cohort.
EXTRACT <- list(GSE31210 = clin_GSE31210,
                GSE68465 = clin_GSE68465,
                GSE72094 = clin_GSE72094)

# ---- main loop: one pass per cohort --------------------------------------
# For each series in the configuration: load the filtered ExpressionSet,
# write the gene-level expression table, write the clinical table, and
# print how many samples have a usable outcome and covariates.
# If the input is missing, the error message says "run `make filter-geo`".
# This repository has no such target: run prep/01b_filter_geo.R, or
# `make prep`, which starts with it.
for (co in cfg$data$geo) {
  id <- co$id
  f  <- file.path(cfg$paths$interim, sprintf("%s_tumour_eset.rds", id))
  if (!file.exists(f)) {
    stop(sprintf("%s not found -- run `Rscript prep/01b_filter_geo.R` first", f))
  }
  cat(sprintf("\n%s  (%s, %s)\n", id, co$platform, co$role))
  e <- readRDS(f)

  # Expression: probes to genes, then a table whose first column is the
  # gene symbol (a parquet file has no row names).
  x <- collapse_probes(e)
  expr <- data.frame(gene_name = rownames(x), x, check.names = FALSE)
  arrow::write_parquet(expr, file.path(out, sprintf("geo_%s_expr.parquet", id)))

  # Clinical table. Its rows must be the expression columns, in the same
  # order; the script stops if they are not.
  cl <- EXTRACT[[id]](pData(e))
  stopifnot(identical(cl$sample_id, colnames(x)))
  arrow::write_parquet(cl, file.path(out, sprintf("geo_%s_clinical.parquet", id)))

  # "Usable" means: a time above zero and a known event status.
  n_os  <- sum(!is.na(cl$os_time) & !is.na(cl$os_event) & cl$os_time > 0)
  n_rfs <- sum(!is.na(cl$rfs_time) & !is.na(cl$rfs_event) & cl$rfs_time > 0)
  cat(sprintf("    OS  usable %3d  events %3d\n", n_os,
              sum(cl$os_event[!is.na(cl$os_time)], na.rm = TRUE)))
  cat(sprintf("    RFS usable %3d  events %3d\n", n_rfs,
              sum(cl$rfs_event[!is.na(cl$rfs_time)], na.rm = TRUE)))
  cat(sprintf("    stage recorded %3d/%d   smoking %3d/%d\n",
              sum(!is.na(cl$stage_ordinal)), nrow(cl),
              sum(!is.na(cl$ever_smoker)), nrow(cl)))
}
cat("\nwrote geo_*_expr.parquet and geo_*_clinical.parquet to", out, "\n")
