#!/usr/bin/env Rscript
# =============================================================
# Make the R library PROJECT-LOCAL with renv.
#
#   Rscript prep/00b_init_renv.R
#
# Run AFTER prep/00_setup_r.R. Packages are not re-downloaded:
# renv hydrates from the existing user library via its cache
# (hardlinks), so this is fast and costs almost no extra disk.
#
# Result:
#   renv/library/...   project-local R library (gitignored)
#   renv.lock          exact versions        (COMMITTED)
#   .Rprofile          auto-activates renv   (COMMITTED)
# =============================================================

options(repos = c(CRAN = "https://cloud.r-project.org"))

if (!requireNamespace("renv", quietly = TRUE)) install.packages("renv")

pkgs <- c(
  "arrow", "yaml", "data.table", "dplyr", "tidyr", "readr", "readxl", "stringr",
  "ggplot2", "survival", "glmnet", "randomForestSRC", "dcurves", "mice", "digest",
  "TCGAbiolinks", "DESeq2", "edgeR", "GEOquery", "maftools",
  "SummarizedExperiment", "biomaRt", "BiocManager"
)

message("[1/3] renv::init(bare = TRUE) - creating project-local library")
renv::init(bare = TRUE, restart = FALSE)

message("\n[2/3] renv::hydrate() - linking packages from the user library")
renv::hydrate(packages = pkgs)

message("\n[3/3] renv::snapshot() - writing renv.lock")
renv::snapshot(packages = pkgs, prompt = FALSE)

message("\nLibrary now: ", paste(.libPaths(), collapse = " | "))
message("Done. Commit renv.lock and .Rprofile; renv/library/ is gitignored.")
