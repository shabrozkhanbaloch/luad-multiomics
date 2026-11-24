#!/usr/bin/env Rscript
# =============================================================
# 01b_filter_geo.R -- keep only the tumour samples of the three GEO cohorts
#
# What it is for
#   The series matrices fetched by 01_download.R contain adjacent normal
#   lung and, in GSE68465, Stratagene reference RNA. Those samples have no
#   survival endpoint and must not reach the external-validation step.
#   This script keeps the tumour samples and counts what it removed.
#
# Where it sits
#   Step 1 of 9 of `make prep`, after `make download`.
#   prep/utils/export_geo.R (step 8) reads what this script writes.
#
# Reads    data/raw/<ID>_eset.rds, one per GEO series. Each is an
#          ExpressionSet: an R object holding the expression table, a
#          description of every sample (called phenoData) and a
#          description of every probe.
#          config/config.yaml, section data: geo: for each series, the
#          phenoData column that says what the tissue is (tumour_field) and
#          the value that means "tumour" (tumour_value). The rule is in the
#          config because the three series write the tissue type in three
#          different columns.
# Writes   data/interim/<ID>_tumour_eset.rds  the same object, tumours only
#          results/tables/table_geo_filtering.csv  samples before and
#                                                  after, per cohort
#          The raw files in data/raw/ are never changed.
#
# How to run (from the project folder)
#   Rscript prep/01b_filter_geo.R              # all cohorts
#   Rscript prep/01b_filter_geo.R GSE31210     # one cohort only
#
# Safe to run again: the output files are overwritten with the same result.
# =============================================================

# yaml reads the configuration; Biobase provides the ExpressionSet object.
suppressPackageStartupMessages({
  library(yaml)
  library(Biobase)
})

# Folders from the configuration: raw data, intermediate files and result
# tables. The seed is set as a precaution; nothing here is random.
cfg     <- yaml::read_yaml("config/config.yaml")
RAW     <- cfg$paths$raw
INTERIM <- cfg$paths$interim
TABLES  <- cfg$paths$tables
set.seed(cfg$project$seed)

dir.create(INTERIM, recursive = TRUE, showWarnings = FALSE)
dir.create(TABLES,  recursive = TRUE, showWarnings = FALSE)

# The cohorts to process: the ones named on the command line, or every
# series listed in the configuration.
args    <- commandArgs(trailingOnly = TRUE)
wanted  <- if (length(args)) args else vapply(cfg$data$geo, `[[`, "", "id")

# Print a message with the time of day in front of it.
log_msg <- function(...) cat(format(Sys.time(), "[%H:%M:%S] "), ..., "\n", sep = "")

# One row of counts per cohort is collected here and written at the end.
summary_rows <- list()

# ---------------------------------------------------------------
# One pass per cohort. `spec` is that cohort's entry in the configuration.
# ---------------------------------------------------------------
for (spec in cfg$data$geo) {
  id <- spec$id
  if (!(id %in% wanted)) next

  # The raw file must exist; if not, the download step has not been run.
  src <- file.path(RAW, paste0(id, "_eset.rds"))
  if (!file.exists(src)) {
    stop("missing raw eset: ", src, "\n  run `make download` first")
  }

  # Load the cohort. `pd` is the sample description: one row per sample.
  # `n_in` is the number of samples before filtering.
  log_msg("=== ", id, " (", spec$role, ") ===")
  eset <- readRDS(src)
  pd   <- Biobase::pData(eset)
  n_in <- ncol(eset)

  # The selection rule for this cohort, from the configuration.
  field <- spec$tumour_field
  value <- spec$tumour_value
  if (is.null(field) || is.null(value)) {
    stop(id, ": config/config.yaml is missing tumour_field / tumour_value")
  }

  # Stop with a clear message if the column is not there. The alternative
  # would be to keep every sample without saying so: a renamed or absent
  # column would then let normal tissue through into external validation.
  if (!(field %in% colnames(pd))) {
    stop(id, ": phenoData has no column '", field, "'\n",
         "  available: ", paste(colnames(pd), collapse = ", "))
  }

  # `keep` is TRUE for the samples whose value is exactly the tumour value.
  # If no sample matches, the value in the configuration is wrong for this
  # series, and the script stops and shows the values it did find.
  observed <- as.character(pd[[field]])
  keep     <- !is.na(observed) & observed == value
  if (!any(keep)) {
    stop(id, ": '", field, "' has no value '", value, "'\n",
         "  observed: ", paste(unique(observed), collapse = " | "))
  }

  # Print the rule and how many samples carry each value, so that the log
  # shows what was kept and what was dropped.
  log_msg("  rule: ", field, " == '", value, "'")
  cat("  observed levels:\n")
  print(table(observed, useNA = "ifany"))

  # Subsetting an ExpressionSet by column keeps the expression values and
  # the sample description together.
  filtered <- eset[, keep]
  n_out    <- ncol(filtered)
  n_drop   <- n_in - n_out

  # Check that the kept samples are the intended ones, in the same order.
  # A mismatch here would attach the wrong clinical record to a sample in
  # every later step.
  stopifnot(n_out == sum(keep),
            identical(Biobase::sampleNames(filtered),
                      Biobase::sampleNames(eset)[keep]))

  # The configuration states how many tumour samples each cohort should
  # have (expected_n). A different number prints a warning; it does not
  # stop the script.
  expected <- spec$expected_n
  if (!is.null(expected) && n_out != expected) {
    log_msg("  !! expected ", expected, " tumour samples, kept ", n_out,
            " -- check the series annotation before using this cohort")
  }

  # Save the filtered copy under a new name in data/interim/.
  dest <- file.path(INTERIM, paste0(id, "_tumour_eset.rds"))
  saveRDS(filtered, dest)
  log_msg("  kept ", n_out, " / ", n_in, "  (dropped ", n_drop, ")")
  log_msg("  wrote ", dest)

  # Remember the counts and the rule for the summary table.
  summary_rows[[id]] <- data.frame(
    cohort       = id,
    platform     = spec$platform,
    role         = spec$role,
    n_raw        = n_in,
    n_tumour     = n_out,
    n_dropped    = n_drop,
    rule         = paste0(field, " == '", value, "'"),
    stringsAsFactors = FALSE
  )
}

# ---------------------------------------------------------------
# Write the summary table: one row per cohort.
# ---------------------------------------------------------------
if (length(summary_rows)) {
  tbl <- do.call(rbind, summary_rows)
  rownames(tbl) <- NULL
  out <- file.path(TABLES, "table_geo_filtering.csv")

  # If only one cohort was processed this time, the table must not shrink
  # to one row. The rows of the other cohorts are taken from the existing
  # file, as long as it has the same columns.
  if (file.exists(out)) {
    prev <- utils::read.csv(out, stringsAsFactors = FALSE)
    if (identical(sort(colnames(prev)), sort(colnames(tbl)))) {
      prev <- prev[!(prev$cohort %in% tbl$cohort), , drop = FALSE]
      tbl  <- rbind(tbl, prev[, colnames(tbl), drop = FALSE])
    } else {
      log_msg("  note: existing ", basename(out),
              " has different columns -- replacing it")
    }
  }

  # Sort the rows in the order of the configuration, so that the table
  # looks the same however it was rebuilt.
  order_ids <- vapply(cfg$data$geo, `[[`, "", "id")
  tbl <- tbl[order(match(tbl$cohort, order_ids)), , drop = FALSE]
  rownames(tbl) <- NULL

  write.csv(tbl, out, row.names = FALSE)
  cat("\n")
  print(tbl)
  log_msg("wrote ", out)
}
