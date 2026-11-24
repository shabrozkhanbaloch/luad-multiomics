#!/usr/bin/env Rscript
# Verify the R toolchain. Exit 1 if anything required is missing.
#   Rscript prep/utils/check_env.R

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
# scikit-survival in Python, so they are NOT required. See README.
optional <- c("timeROC", "pec", "riskRegression")

show <- function(p) {
  has <- requireNamespace(p, quietly = TRUE)
  v <- if (has) as.character(utils::packageVersion(p)) else "--"
  cat(sprintf("  %-22s %-10s %s\n", p, v, if (has) "OK" else "MISSING"))
  has
}

cat("  R", paste0(R.version$major, ".", R.version$minor), "\n")

ok <- vapply(required, show, logical(1))

cat("\n  optional (Python provides these - see README):\n")
invisible(vapply(optional, show, logical(1)))

if (!all(ok)) {
  cat("\n  MISSING (required):", paste(required[!ok], collapse = ", "), "\n")
  cat("  Fix with: Rscript prep/00_setup_r.R\n")
  quit(status = 1)
}

cat(sprintf("\n  all %d required R packages OK\n", length(required)))
