#!/usr/bin/env Rscript
# =============================================================
# check_env.R -- check that the R packages needed by prep/ are installed
#
# What it is for
#   Prints one line per package: its name, its version, and OK or MISSING.
#   It ends with an error status (1) if a required package is missing.
#
# Where it sits
#   After prep/00_setup_r.R and before `make download` and `make prep`.
#   Run it by hand; it is not part of any Makefile target.
#
# Reads    nothing from the project.
# Writes   nothing; it only prints.
#
# How to run (from the project folder)
#   Rscript prep/utils/check_env.R
# =============================================================

# The packages that must be present. The same list as in prep/00_setup_r.R.
required <- c(
  # CRAN
  "arrow", "yaml", "data.table", "dplyr", "tidyr", "readr", "readxl", "stringr",
  "ggplot2", "survival", "glmnet", "randomForestSRC", "dcurves", "mice", "digest",
  # Bioconductor
  "TCGAbiolinks", "DESeq2", "edgeR", "GEOquery", "maftools",
  "SummarizedExperiment", "biomaRt"
)

# Not installable on R < 4.4 (chain: timeROC -> pec -> riskRegression -> rms,
# and rms >= 8.1 requires R >= 4.4.0). Their functionality is provided by
# scikit-survival in Python, so they are NOT required. They are listed only
# so that you can see whether you happen to have them. The longer
# explanation is in prep/00_setup_r.R.
optional <- c("timeROC", "pec", "riskRegression")

# Helper: print one line for package `p` and return TRUE if it is installed.
show <- function(p) {
  has <- requireNamespace(p, quietly = TRUE)
  v <- if (has) as.character(utils::packageVersion(p)) else "--"
  cat(sprintf("  %-22s %-10s %s\n", p, v, if (has) "OK" else "MISSING"))
  has
}

cat("  R", paste0(R.version$major, ".", R.version$minor), "\n")

# Check the required packages. `ok` holds TRUE or FALSE for each one.
ok <- vapply(required, show, logical(1))

cat("\n  optional (Python provides these - see README):\n")
# Show the optional ones for information; the result is not used.
invisible(vapply(optional, show, logical(1)))

# Name what is missing, say how to fix it, and stop with an error status.
if (!all(ok)) {
  cat("\n  MISSING (required):", paste(required[!ok], collapse = ", "), "\n")
  cat("  Fix with: Rscript prep/00_setup_r.R\n")
  quit(status = 1)
}

cat(sprintf("\n  all %d required R packages OK\n", length(required)))
