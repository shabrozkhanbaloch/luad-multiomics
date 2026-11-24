#!/usr/bin/env Rscript
# export_geo.R -- turn the three filtered GEO ExpressionSets into the two
# parquet files 11_external_validation.py needs, per cohort:
#
#   data/processed/geo_<ID>_expr.parquet      gene_name x sample, one row per gene
#   data/processed/geo_<ID>_clinical.parquet  outcomes and covariates
#
# Three cohorts, three platforms, three annotation vocabularies. Every
# decision below is about reconciling them WITHOUT quietly inventing
# agreement that is not there.
#
# Probe -> gene symbol
#   A symbol measured by several probes keeps the probe with the highest
#   mean intensity, which is the standard collapse rule and favours the
#   probe with the best dynamic range. Probes annotated to several symbols
#   ("DDR1 /// MIR4640") are DROPPED rather than assigned to the first:
#   an ambiguous probe is not evidence about either gene.
#
# Expression scale
#   GSE31210 and GSE68465 arrive on a linear intensity scale, GSE72094 on
#   log2. Nothing is done about that here ON PURPOSE. The model consumes
#   a within-sample RANK, and a rank is invariant to any monotone
#   per-sample transform, so log and linear give the identical
#   representation. That is the reason the protocol chose the rank, and
#   11 asserts it rather than trusting this comment.
#
# Endpoints
#   The only endpoint all three cohorts record is overall survival. TCGA's
#   primary endpoint is PFI, so 11 trains its transferring models on
#   TCGA's OS -- the config's declared sensitivity endpoint -- to compare
#   like with like. GSE31210 also records relapse-free survival, which is
#   the closer analogue of PFI, and it is exported so 11 can run that as a
#   secondary check on the one cohort that allows it.
suppressPackageStartupMessages({
  library(Biobase); library(arrow); library(yaml)
})

cfg <- yaml::read_yaml("config/config.yaml")
out <- cfg$paths$processed
dir.create(out, recursive = TRUE, showWarnings = FALSE)

`%||%` <- function(a, b) if (is.null(a) || length(a) == 0) b else a

num <- function(x) suppressWarnings(as.numeric(trimws(as.character(x))))

# GEO writes missingness as "--", "NA", "Unknown", "Missing" and "".
blank <- function(x) {
  x <- trimws(as.character(x))
  x[x %in% c("--", "NA", "", "Unknown", "unknown", "Missing", "missing")] <- NA
  x
}

symbol_column <- function(fd) {
  hit <- intersect(c("Gene Symbol", "GeneSymbol", "SYMBOL", "gene_symbol"),
                   colnames(fd))
  if (!length(hit)) stop("no gene-symbol column in featureData")
  hit[1]
}

collapse_probes <- function(e) {
  fd <- fData(e)
  sym <- blank(fd[[symbol_column(fd)]])
  x   <- exprs(e)

  # Multi-mapping probes carry no unambiguous evidence about any one gene.
  ambiguous <- grepl("///", sym, fixed = TRUE)
  drop <- is.na(sym) | sym == "" | ambiguous
  cat(sprintf("    probes %6d  ->  dropped %5d (%d unannotated, %d multi-symbol)\n",
              nrow(x), sum(drop), sum(is.na(sym) | sym == ""), sum(ambiguous)))
  x <- x[!drop, , drop = FALSE]; sym <- sym[!drop]

  # Highest mean intensity wins the symbol.
  ord  <- order(rowMeans(x, na.rm = TRUE), decreasing = TRUE)
  x    <- x[ord, , drop = FALSE]; sym <- sym[ord]
  keep <- !duplicated(sym)
  x    <- x[keep, , drop = FALSE]
  rownames(x) <- sym[keep]
  cat(sprintf("    genes  %6d after collapsing to symbols\n", nrow(x)))
  x[order(rownames(x)), , drop = FALSE]
}

# ---- per-cohort outcome and covariate extraction ------------------------
# Deliberately one function per cohort. A generic mapper over three
# incompatible vocabularies would be shorter and would hide exactly the
# decisions a reader needs to check.

clin_GSE31210 <- function(pd) {
  # Days, and a two-level status field, for both endpoints.
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
  prog <- blank(pd[["first_progression_or_relapse:ch1"]])
  d <- data.frame(
    sample_id = rownames(pd),
    os_time   = num(pd[["months_to_last_contact_or_death:ch1"]]),
    os_event  = as.integer(blank(pd[["vital_status:ch1"]]) == "Dead"),
    # Recorded only for the patients who progressed; the rest are censored
    # at last contact. Reconstructed rather than taken as given.
    rfs_time  = ifelse(!is.na(prog) & prog == "Yes",
                       num(pd[["months_to_first_progression:ch1"]]),
                       num(pd[["months_to_last_contact_or_death:ch1"]])),
    rfs_event = ifelse(is.na(prog), NA_integer_, as.integer(prog == "Yes")),
    age       = num(pd[["age:ch1"]]),
    sex_male  = as.integer(blank(pd[["Sex:ch1"]]) == "Male"),
    stringsAsFactors = FALSE)
  # Stage is recorded as pNxpTy, not as an AJCC stage group. Mapping it to
  # one properly needs M status, which this cohort does not report, so a
  # coarse node-and-size ordinal is used and named as such.
  st <- blank(pd[["disease_stage:ch1"]])
  n  <- suppressWarnings(as.integer(sub(".*pN([0-9]).*", "\\1", st)))
  t_ <- suppressWarnings(as.integer(sub(".*pT([0-9]).*", "\\1", st)))
  d$stage_ordinal <- ifelse(is.na(n) | is.na(t_), NA,
                     ifelse(n == 0 & t_ <= 1, 1,
                     ifelse(n == 0,           2,
                     ifelse(n == 1,           2, 3))))
  sm <- blank(pd[["smoking_history:ch1"]])
  d$ever_smoker <- ifelse(is.na(sm), NA_integer_,
                          as.integer(sm != "Never smoked"))
  d
}

clin_GSE72094 <- function(pd) {
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

EXTRACT <- list(GSE31210 = clin_GSE31210,
                GSE68465 = clin_GSE68465,
                GSE72094 = clin_GSE72094)

for (co in cfg$data$geo) {
  id <- co$id
  f  <- file.path(cfg$paths$interim, sprintf("%s_tumour_eset.rds", id))
  if (!file.exists(f)) {
    stop(sprintf("%s not found -- run `make filter-geo` first", f))
  }
  cat(sprintf("\n%s  (%s, %s)\n", id, co$platform, co$role))
  e <- readRDS(f)

  x <- collapse_probes(e)
  expr <- data.frame(gene_name = rownames(x), x, check.names = FALSE)
  arrow::write_parquet(expr, file.path(out, sprintf("geo_%s_expr.parquet", id)))

  cl <- EXTRACT[[id]](pData(e))
  stopifnot(identical(cl$sample_id, colnames(x)))
  arrow::write_parquet(cl, file.path(out, sprintf("geo_%s_clinical.parquet", id)))

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
