#!/usr/bin/env Rscript
# =============================================================
# 00_setup_r.R -- install the R packages that the prep/ scripts need
#
# What it is for
#   The data-preparation scripts written in R use packages from CRAN (the
#   general archive of R packages) and from Bioconductor (the archive of
#   packages for biology). This script installs every one that is missing.
#
# Where it sits
#   Before everything else, and only for people who rebuild the data from
#   the public sources (`make download`, `make prep`). It is not part of any
#   Makefile target: run it by hand, once. If you downloaded the prepared
#   data with `make data`, you do not need R at all.
#
# Reads    nothing from the project. It needs the internet.
# Writes   packages into your R library, and results/sessionInfo.txt
#          (the R version and the package versions of this installation).
#
# How heavy
#   Compiles several Bioconductor packages: expect 30-60 minutes on
#   first run. Run this EARLY - it is the longest setup step.
#   Re-running is cheap: already-installed packages are skipped.
#
# How to run (from the project folder)
#   Rscript prep/00_setup_r.R
#
# Afterwards
#   Rscript prep/utils/check_env.R    lists every package and its version
#   Rscript prep/00b_init_renv.R      optional: a library inside the project
# =============================================================

# Settings for the installation:
#   repos    where CRAN packages are downloaded from
#   Ncpus    how many processor cores compile packages at once
#            (all but two, and never fewer than one)
#   timeout  seconds allowed for one download. R's default is 60, which is
#            too short for large packages; this allows 20 minutes.
options(
  repos  = c(CRAN = "https://cloud.r-project.org"),
  Ncpus  = max(1L, parallel::detectCores() - 2L),
  timeout = 1200
)

message("R version : ", R.version.string)
message("Compiling with ", getOption("Ncpus"), " cores")

# The first library path is where R installs new packages. It is printed so
# that you can see which folder is being filled.
lib <- .libPaths()[1]
message("Library   : ", lib)

# Helper used by every step below. It takes a list of package names and the
# function that installs them, and installs only the packages that cannot be
# loaded yet. This is why a second run is quick.
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
# renv records package versions (see prep/00b_init_renv.R).
# BiocManager is the installer for Bioconductor packages.
install_if_missing(c("renv", "BiocManager"), function(p) install.packages(p))

# Each Bioconductor release works with one version of R. Asking for
# BiocManager::version() selects the release that matches the R you are
# running, so that the packages fit together. `update = FALSE` leaves the
# packages you already have as they are.
BiocManager::install(version = BiocManager::version(), ask = FALSE, update = FALSE)
message("  Bioconductor ", as.character(BiocManager::version()))

# ---- 2. CRAN packages ------------------------------------------------------
message("\n[2/4] CRAN packages")
# The note beside each package says what it is for.
# This list was carried over from the earlier project (README, "Things to
# know"), where R was also used for modelling. The prep/ scripts of this
# repository load only yaml, arrow, data.table, readxl and digest from it;
# the models of this project are fitted in Python.
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

# NOT installed: timeROC, pec and riskRegression (survival metrics in R).
# They depend on each other in a chain: timeROC -> pec -> riskRegression ->
# rms, and rms >= 8.1 requires R >= 4.4.0. The R version recorded in
# renv.lock is 4.3.3, which goes with Bioconductor 3.18.
# Upgrading R purely for these would force a full Bioconductor rebuild.
# They are not needed: the same measures exist in the Python package
# scikit-survival, which the analysis uses:
#   timeROC        -> sksurv.metrics.cumulative_dynamic_auc
#   pec (IBS)      -> sksurv.metrics.integrated_brier_score
#   riskRegression -> sksurv.metrics.concordance_index_ipcw + calibration

# ---- 3. Bioconductor packages ---------------------------------------------
message("\n[3/4] Bioconductor packages (this is the slow part)")
# The prep/ scripts of this repository load TCGAbiolinks, GEOquery, maftools
# and SummarizedExperiment. DESeq2, edgeR and biomaRt were carried over with
# the list and are not loaded by any script here.
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
# A package can fail to compile without stopping the script. So every package
# is tested once more here, and the ones that still cannot be loaded are named.
all_pkgs <- c(cran_pkgs, bioc_pkgs)
ok  <- vapply(all_pkgs, requireNamespace, logical(1), quietly = TRUE)

if (all(ok)) {
  message("  ALL ", length(all_pkgs), " PACKAGES OK")
} else {
  message("  FAILED: ", paste(all_pkgs[!ok], collapse = ", "))
}

# Keep a record of this installation: R version, operating system and the
# version of every loaded package.
dir.create("results", showWarnings = FALSE)
writeLines(capture.output(sessionInfo()), "results/sessionInfo.txt")
message("  session info -> results/sessionInfo.txt")

# renv.lock is a file that lists the exact version of every package, so that
# someone else can install the same ones. The lines below only print a
# reminder; prep/00b_init_renv.R does these two commands for you.
if (requireNamespace("renv", quietly = TRUE)) {
  message("\n  To lock these versions, run in an R session at the project root:")
  message("    renv::init(bare = TRUE); renv::snapshot()")
}

# End with an error status if a package is missing, so that the problem is
# not overlooked.
if (!all(ok)) quit(status = 1)
message("\nDone.")
