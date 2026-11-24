#!/usr/bin/env Rscript
# =============================================================
# 01b_filter_geo.R  -- drop non-tumour samples from the GEO cohorts
#
#   Rscript prep/01b_filter_geo.R              # all cohorts
#   Rscript prep/01b_filter_geo.R GSE31210     # one cohort only
#
# The series matrices fetched by 01_download.R contain adjacent normal
# lung and, in GSE68465, Stratagene reference RNA. Those samples have no
# survival endpoint and must not reach the external-validation step.
#
# Raw esets are never modified. Filtered copies are written to
# data/interim/<id>_tumour_eset.rds and the counts to
# results/tables/table_geo_filtering.csv.
#
# The selection rule for each cohort lives in config/config.yaml
# (tumour_field / tumour_value) because the three series annotate
# tissue type in three different phenoData columns.
#
# Idempotent: re-running overwrites the interim files with the same result.
# =============================================================

suppressPackageStartupMessages({
  library(yaml)
  library(Biobase)
})

cfg     <- yaml::read_yaml("config/config.yaml")
RAW     <- cfg$paths$raw
INTERIM <- cfg$paths$interim
TABLES  <- cfg$paths$tables
set.seed(cfg$project$seed)

dir.create(INTERIM, recursive = TRUE, showWarnings = FALSE)
dir.create(TABLES,  recursive = TRUE, showWarnings = FALSE)

args    <- commandArgs(trailingOnly = TRUE)
wanted  <- if (length(args)) args else vapply(cfg$data$geo, `[[`, "", "id")

log_msg <- function(...) cat(format(Sys.time(), "[%H:%M:%S] "), ..., "\n", sep = "")

summary_rows <- list()

for (spec in cfg$data$geo) {
  id <- spec$id
  if (!(id %in% wanted)) next

  src <- file.path(RAW, paste0(id, "_eset.rds"))
  if (!file.exists(src)) {
    stop("missing raw eset: ", src, "\n  run `make download` first")
  }

  log_msg("=== ", id, " (", spec$role, ") ===")
  eset <- readRDS(src)
  pd   <- Biobase::pData(eset)
  n_in <- ncol(eset)

  field <- spec$tumour_field
  value <- spec$tumour_value
  if (is.null(field) || is.null(value)) {
    stop(id, ": config/config.yaml is missing tumour_field / tumour_value")
  }

  # Fail loudly rather than silently keeping every sample: a renamed or
  # absent phenoData column would otherwise pass normals straight through
  # into external validation.
  if (!(field %in% colnames(pd))) {
    stop(id, ": phenoData has no column '", field, "'\n",
         "  available: ", paste(colnames(pd), collapse = ", "))
  }

  observed <- as.character(pd[[field]])
  keep     <- !is.na(observed) & observed == value
  if (!any(keep)) {
    stop(id, ": '", field, "' has no value '", value, "'\n",
         "  observed: ", paste(unique(observed), collapse = " | "))
  }

  log_msg("  rule: ", field, " == '", value, "'")
  cat("  observed levels:\n")
  print(table(observed, useNA = "ifany"))

  filtered <- eset[, keep]
  n_out    <- ncol(filtered)
  n_drop   <- n_in - n_out

  # Sample identity must survive the subset; a silent mismatch here would
  # misalign every downstream clinical join.
  stopifnot(n_out == sum(keep),
            identical(Biobase::sampleNames(filtered),
                      Biobase::sampleNames(eset)[keep]))

  expected <- spec$expected_n
  if (!is.null(expected) && n_out != expected) {
    log_msg("  !! expected ", expected, " tumour samples, kept ", n_out,
            " -- check the series annotation before using this cohort")
  }

  dest <- file.path(INTERIM, paste0(id, "_tumour_eset.rds"))
  saveRDS(filtered, dest)
  log_msg("  kept ", n_out, " / ", n_in, "  (dropped ", n_drop, ")")
  log_msg("  wrote ", dest)

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

if (length(summary_rows)) {
  tbl <- do.call(rbind, summary_rows)
  rownames(tbl) <- NULL
  out <- file.path(TABLES, "table_geo_filtering.csv")

  # A single-cohort run must not shrink the table to one row: carry over the
  # rows for cohorts that were not rebuilt this time.
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

  # Keep config order so the table reads the same however it was rebuilt.
  order_ids <- vapply(cfg$data$geo, `[[`, "", "id")
  tbl <- tbl[order(match(tbl$cohort, order_ids)), , drop = FALSE]
  rownames(tbl) <- NULL

  write.csv(tbl, out, row.names = FALSE)
  cat("\n")
  print(tbl)
  log_msg("wrote ", out)
}
