#!/usr/bin/env Rscript
# =============================================================
# R / Bioconductor environment setup
#
#   Rscript prep/00_setup_r.R
#
# Compiles several Bioconductor packages: expect 30-60 minutes on
# first run. Run this EARLY - it is the longest setup step.
# Re-running is cheap: already-installed packages are skipped.
# =============================================================

options(
  repos  = c(CRAN = "https://cloud.r-project.org"),
  Ncpus  = max(1L, parallel::detectCores() - 2L),
  timeout = 1200
)

message("R version : ", R.version.string)
message("Compiling with ", getOption("Ncpus"), " cores")

lib <- .libPaths()[1]
message("Library   : ", lib)

install_if_missing <- function(pkgs, installer) {
  missing <- pkgs[!vapply(pkgs, requireNamespace, logical(1), quietly = TRUE)]
  if (!length(missing)) {
    message("  all present: ", paste(pkgs, collapse = ", "))
    return(invisible(NULL))
  }
  message("  installing: ", paste(missing, collapse = ", "))
  installer(missing)
}

# ---- 1. renv + BiocManager -------------------------------------------------
message("\n[1/4] renv and BiocManager")
install_if_missing(c("renv", "BiocManager"), function(p) install.packages(p))

# Bioconductor release matched to this R version
BiocManager::install(version = BiocManager::version(), ask = FALSE, update = FALSE)
message("  Bioconductor ", as.character(BiocManager::version()))

# ---- 2. CRAN packages ------------------------------------------------------
message("\n[2/4] CRAN packages")
cran_pkgs <- c(
  "arrow",             # Parquet handoff with Python
  "yaml",              # reads config/config.yaml
  "data.table",
  "dplyr", "tidyr", "readr", "readxl", "stringr",
  "ggplot2",
  "survival",          # Cox, Schoenfeld residuals
  "glmnet",            # elastic-net Cox
  "randomForestSRC",   # random survival forests
  "dcurves",           # decision curve analysis
  "mice",              # multiple imputation
  "digest"             # checksums for the data manifest
)
install_if_missing(cran_pkgs, function(p) install.packages(p))

# NOT installed: timeROC / pec / riskRegression.
# Chain: timeROC -> pec -> riskRegression -> rms, and rms >= 8.1 requires
# R >= 4.4.0 (this system runs R 4.3.3, tied to Bioconductor 3.18).
# All of their functionality is provided by scikit-survival in Python:
#   timeROC        -> sksurv.metrics.cumulative_dynamic_auc
#   pec (IBS)      -> sksurv.metrics.integrated_brier_score
#   riskRegression -> sksurv.metrics.concordance_index_ipcw + calibration
# Upgrading R purely for these would force a full Bioconductor rebuild.

# ---- 3. Bioconductor packages ---------------------------------------------
message("\n[3/4] Bioconductor packages (this is the slow part)")
bioc_pkgs <- c(
  "TCGAbiolinks",      # GDC download
  "DESeq2",            # variance-stabilising transform
  "edgeR",             # filterByExpr
  "GEOquery",          # external cohorts
  "maftools",          # MC3 MAF handling
  "SummarizedExperiment",
  "biomaRt"            # gene annotation / platform mapping
)
install_if_missing(bioc_pkgs, function(p) BiocManager::install(p, ask = FALSE, update = FALSE))

# ---- 4. Verify and record --------------------------------------------------
message("\n[4/4] Verification")
all_pkgs <- c(cran_pkgs, bioc_pkgs)
ok  <- vapply(all_pkgs, requireNamespace, logical(1), quietly = TRUE)

if (all(ok)) {
  message("  ALL ", length(all_pkgs), " PACKAGES OK")
} else {
  message("  FAILED: ", paste(all_pkgs[!ok], collapse = ", "))
}

dir.create("results", showWarnings = FALSE)
writeLines(capture.output(sessionInfo()), "results/sessionInfo.txt")
message("  session info -> results/sessionInfo.txt")

# renv.lock records exact versions for reproducibility
if (requireNamespace("renv", quietly = TRUE)) {
  message("\n  To lock these versions, run in an R session at the project root:")
  message("    renv::init(bare = TRUE); renv::snapshot()")
}

if (!all(ok)) quit(status = 1)
message("\nDone.")
