#!/usr/bin/env Rscript
# =============================================================
# 00b_init_renv.R -- give the project its own R library with renv (optional)
#
# What it is for
#   renv is an R package that keeps a separate package library inside the
#   project folder and writes the exact version of every package to the
#   file renv.lock. With that file, someone else can install the same
#   versions (renv::restore()).
#
# Where it sits
#   After prep/00_setup_r.R. Optional, run by hand, once. It is not part of
#   any Makefile target, and the prep/ scripts work without it.
#
# Reads    the R packages that prep/00_setup_r.R installed.
# Writes   renv/library/...   the project's own R library (renv keeps it
#                             out of git with its own renv/.gitignore)
#          renv.lock          exact versions (this file is in the repository)
#          .Rprofile          a start-up file that switches renv on each time
#                             R starts in this folder. It is not in the
#                             repository at present; this script creates it.
#
# How heavy
#   Packages are not re-downloaded: renv hydrates from the existing user
#   library via its cache (hardlinks), so this is fast and costs almost no
#   extra disk.
#
# How to run (from the project folder)
#   Rscript prep/00b_init_renv.R
#
# Good to know: after this script, R started in this folder uses the
# project library and no longer your personal one.
# =============================================================

# Where to download from, if anything has to be downloaded at all.
options(repos = c(CRAN = "https://cloud.r-project.org"))

# Install renv itself if prep/00_setup_r.R has not been run.
if (!requireNamespace("renv", quietly = TRUE)) install.packages("renv")

# The packages to record. It is the list of prep/00_setup_r.R: the CRAN
# packages, then the Bioconductor packages, then BiocManager. If you change
# one list, change the other, and prep/utils/check_env.R as well.
pkgs <- c(
  "arrow", "yaml", "data.table", "dplyr", "tidyr", "readr", "readxl", "stringr",
  "ggplot2", "survival", "glmnet", "randomForestSRC", "dcurves", "mice", "digest",
  "TCGAbiolinks", "DESeq2", "edgeR", "GEOquery", "maftools",
  "SummarizedExperiment", "biomaRt", "BiocManager"
)

# Step 1. `bare = TRUE` creates an empty project library without searching
# the code for packages. `restart = FALSE` keeps this R session running, so
# that the next two steps can follow.
message("[1/3] renv::init(bare = TRUE) - creating project-local library")
renv::init(bare = TRUE, restart = FALSE)

# Step 2. "Hydrate" means: link the listed packages, and the packages they
# depend on, from the library you already have into the project library.
message("\n[2/3] renv::hydrate() - linking packages from the user library")
renv::hydrate(packages = pkgs)

# Step 3. Write the names and versions to renv.lock. `prompt = FALSE` means
# no question is asked before the file is written.
message("\n[3/3] renv::snapshot() - writing renv.lock")
renv::snapshot(packages = pkgs, prompt = FALSE)

# Show which library folders R uses now. The project library comes first.
message("\nLibrary now: ", paste(.libPaths(), collapse = " | "))
message("Done. Commit renv.lock and .Rprofile; renv/library/ is gitignored.")
