#!/usr/bin/env Rscript
# =============================================================
# 01_download.R -- fetch the raw public data
#
# What it is for
#   Downloads the public sources this project is built on and saves them
#   in data/raw/. It is the first step of rebuilding the data from scratch;
#   `make download` runs it. You do not need it if you downloaded the
#   prepared data with `make data`.
#
# The five blocks, in the order they run
#   tcga      TCGA-LUAD RNA-seq gene counts from the Genomic Data Commons
#             (GDC)                        -> tcga_luad_rnaseq_se.rds
#   clinical  clinical covariates, taken out of the tcga file (no download)
#                                          -> tcga_luad_clinical.rds
#   cdr       the TCGA Clinical Data Resource: the survival endpoints
#                                          -> TCGA-CDR-SupplementalTableS1.xlsx
#   mc3       the MC3 somatic mutation calls, all TCGA cancers in one file
#                                          -> mc3.v0.2.8.PUBLIC.maf.gz
#   geo       three GEO series, the external cohorts
#                                          -> GSE72094_eset.rds and two more
#
# Not fetched here
#   The DNA methylation files have their own target, `make
#   download-methylation`. The probe annotation file that
#   prep/16_preprocess_methylation.R also needs
#   (data/raw/methylation/HM450.hg38.manifest.tsv.gz) is not fetched by any
#   script of this repository.
#
# Reads    config/config.yaml: the two download addresses (data: urls:),
#          the TCGA project and workflow, and the list of GEO series.
# Writes   the files above into data/raw/, one .md5 checksum file beside
#          each, and one row per file in data/raw/MANIFEST.md.
#          In this repository the whole data/ folder is ignored by git, so
#          all of this stays on your computer. (The heading that this script
#          writes into MANIFEST.md says the manifest is committed; that was
#          the arrangement of the earlier project.)
#
# How heavy
#   It needs the internet and the R packages of prep/00_setup_r.R. The MC3
#   file alone is about 1 GB. The README gives about 10 GB for the download
#   and the preparation together.
#
# How to run (from the project folder)
#   make download                           # the same as the first line below
#   Rscript prep/01_download.R              # everything
#   Rscript prep/01_download.R tcga         # one block only
#   Rscript prep/01_download.R clinical geo
#
# Safe to run again: anything already downloaded is skipped.
# A block that fails is reported and the other blocks still run. The script
# then ends with an error status and says which blocks to run again.
# =============================================================

# yaml reads the configuration file; digest computes checksums. The larger
# packages (TCGAbiolinks, GEOquery) are loaded only inside the block that
# needs them, so that one block can run without the others' packages.
suppressPackageStartupMessages({
  library(yaml)
  library(digest)
})

# Large public files (MC3 is ~1 GB). R's default download timeout is 60
# seconds, which cuts a large file short without an error. Two hours are
# allowed here, once, for every block.
options(timeout = 7200)

# Read the plan. RAW is the folder for raw data (data/raw). The seed is set
# as a precaution; nothing in this script draws random numbers.
cfg <- yaml::read_yaml("config/config.yaml")
RAW <- cfg$paths$raw
set.seed(cfg$project$seed)
dir.create(RAW, recursive = TRUE, showWarnings = FALSE)

# Which blocks to run: the ones named on the command line, or all five. The
# order matters in one place: `clinical` reads the file that `tcga` writes.
args   <- commandArgs(trailingOnly = TRUE)
blocks <- if (length(args)) args else c("tcga", "clinical", "cdr", "mc3", "geo")

# Print a message with the time of day in front of it.
log_msg <- function(...) cat(format(Sys.time(), "[%H:%M:%S] "), ..., "\n", sep = "")

# The names of the blocks (and GEO series) that failed. Filled in below and
# reported at the end.
failures <- character()

# run_block() runs one block, but only if it was asked for, and catches any
# error. One failing block must not abort the others: without this, an error
# in an early block would stop the script before GEO, the last block, was
# reached.
run_block <- function(name, expr) {
  if (!(name %in% blocks)) return(invisible(NULL))
  tryCatch(force(expr), error = function(e) {
    failures <<- c(failures, name)
    log_msg("!! BLOCK FAILED: ", name)
    message("   ", conditionMessage(e))
    message("   Continuing with the remaining blocks.")
  })
}

# ---- manifest helpers ------------------------------------------------------
# A checksum (here: md5) is a short fingerprint of a file's content. If one
# byte of the file changes, the fingerprint changes. record() computes it for
# a downloaded file, then either compares it with a stored fingerprint or
# stores it. The manifest (data/raw/MANIFEST.md) is the list of all files
# with their description, size and fingerprint.
record <- function(path, description) {
  if (!file.exists(path)) return(invisible(NULL))
  md5  <- digest::digest(path, algo = "md5", file = TRUE)
  size <- file.info(path)$size
  md5_file <- file.path(RAW, paste0(basename(path), ".md5"))

  # A .md5 file that is already there is the EXPECTED fingerprint, not a
  # note to overwrite. Overwriting it with whatever was just downloaded
  # would hide the one thing the manifest exists to show: that a source
  # file has changed since the results were computed. So: compare when
  # there is something to compare with, and record only when there is not.
  # A mismatch stops the block and prints what to do.
  if (file.exists(md5_file)) {
    expected <- sub("\\s.*$", "", readLines(md5_file, warn = FALSE)[1])
    if (nzchar(expected) && !identical(expected, md5)) {
      stop("CHECKSUM MISMATCH for ", basename(path), "\n",
           "  expected (from the repository): ", expected, "\n",
           "  got (this download)          : ", md5, "\n\n",
           "  The upstream file has changed, or this download is corrupt.\n",
           "  Anything computed from it will not reproduce the committed results.\n\n",
           "  If the new file is the one you want, that is a deliberate change of\n",
           "  input data: delete ", md5_file, ", remove its row from\n",
           "  ", file.path(RAW, "MANIFEST.md"), ", re-run, and commit the new values.")
    }
    log_msg("  checksum matches the repository manifest")
    return(invisible(NULL))
  }

  # First time this file is seen: store its fingerprint, and add a row with
  # its name, description, size and fingerprint to the manifest.
  writeLines(sprintf("%s  %s", md5, basename(path)), md5_file)
  cat(sprintf("| `%s` | %s | %.1f MB | `%s` |\n",
              basename(path), description, size / 1e6, md5),
      file = file.path(RAW, "MANIFEST.md"), append = TRUE)
  log_msg("  recorded ", basename(path), "  (", round(size / 1e6, 1), " MB)")
}

# Create MANIFEST.md with its title and table heading, unless it exists.
manifest_header <- function() {
  mf <- file.path(RAW, "MANIFEST.md")
  if (file.exists(mf)) return(invisible(NULL))
  writeLines(c(
    "# Raw data manifest", "",
    "Generated by `prep/01_download.R`. Raw files are gitignored;",
    "this manifest and the `.md5` files are committed so the exact inputs",
    "can be identified and re-fetched.", "",
    "| File | Description | Size | md5 |", "|---|---|---|---|"
  ), mf)
}
manifest_header()

# A file that exists is not necessarily complete: a download killed by a
# timeout leaves a partial file that looks perfectly real. Every fetch is
# therefore written to a .part file first, checked, and only then renamed.
# validate_file() is that check:
#   any file   must exist and must not be empty
#   .gz        must pass `gzip -t` (needs the gzip program on the computer)
#   .xlsx      must start with the two bytes "PK", as every zip archive does
validate_file <- function(path) {
  if (!file.exists(path) || file.info(path)$size == 0) return(FALSE)
  if (grepl("\\.gz$", path)) {
    return(system2("gzip", c("-t", shQuote(path)),
                   stdout = FALSE, stderr = FALSE) == 0L)
  }
  if (grepl("\\.xlsx$", path)) {          # xlsx is a zip archive
    sig <- readBin(path, "raw", n = 2)
    return(length(sig) == 2 && sig[1] == as.raw(0x50) && sig[2] == as.raw(0x4b))
  }
  TRUE
}

# fetch_url() downloads one file whose address is in config/config.yaml
# under data: urls: <key>. In order:
#   - the file is there and valid: skip it (and record a checksum if none)
#   - the file is there but damaged: delete it and download again
#   - no address is configured: say where to get the file by hand
#   - otherwise: download to <file>.part, check it, rename it, record it
fetch_url <- function(key, dest, description, where_to_get) {
  if (file.exists(dest)) {
    if (!validate_file(dest)) {
      log_msg("!! ", basename(dest), " exists but FAILS integrity check - removing")
      unlink(dest); unlink(file.path(RAW, paste0(basename(dest), ".md5")))
    } else {
      log_msg(basename(dest), " already present and valid - skipping download")
      if (!file.exists(file.path(RAW, paste0(basename(dest), ".md5")))) {
        log_msg("  no checksum yet - recording provenance for the existing file")
        record(dest, description)
      }
      return(invisible(TRUE))
    }
  }

  url <- cfg$data$urls[[key]]
  if (is.null(url) || !nzchar(url) || grepl("^SET_ME", url)) {
    log_msg("SKIPPED: ", basename(dest))
    message("  No URL configured for `data.urls.", key, "` in config/config.yaml.")
    message("  Obtain it from: ", where_to_get)
    message("  Then set the URL in config.yaml (or drop the file into ", dest, ") and re-run.")
    return(invisible(FALSE))
  }

  # on.exit() removes the .part file when the function ends for any reason,
  # so that a failed download leaves nothing behind.
  part <- paste0(dest, ".part")
  on.exit(unlink(part), add = TRUE)
  log_msg("downloading ", basename(dest), " ...")
  utils::download.file(url, part, mode = "wb", quiet = FALSE)

  if (!validate_file(part)) {
    stop("download of ", basename(dest), " failed its integrity check ",
         "(likely truncated). Nothing was written to ", dest)
  }
  file.rename(part, dest)
  log_msg("  integrity check passed")
  record(dest, description)
  invisible(TRUE)
}

# =============================================================
# 1. TCGA-LUAD RNA-seq (STAR counts)
#
# TCGAbiolinks asks the GDC for every gene-expression file of the project,
# downloads them into data/raw/GDCdata, and GDCprepare() joins them into one
# SummarizedExperiment. That is an R object holding the tables of numbers
# (genes in rows, samples in columns) together with a description of every
# gene and every sample. "STAR - Counts" is the GDC's name for its workflow:
# the reads are aligned with the program STAR and counted per gene.
# Every sample of the project is downloaded, normal tissue included. The
# choice of samples is made later, in the prep scripts, where it is logged.
# =============================================================
run_block("tcga", {
  log_msg("=== TCGA-LUAD RNA-seq (STAR - Counts) ===")
  out <- file.path(RAW, "tcga_luad_rnaseq_se.rds")
  if (file.exists(out)) {
    log_msg("  already present - skipping")
  } else {
    suppressPackageStartupMessages(library(TCGAbiolinks))
    q <- GDCquery(project = cfg$data$tcga$project,
                  data.category = "Transcriptome Profiling",
                  data.type     = "Gene Expression Quantification",
                  workflow.type = cfg$data$tcga$workflow)
    log_msg("  query returned ", nrow(getResults(q)), " files")
    GDCdownload(q, directory = file.path(RAW, "GDCdata"))
    se <- GDCprepare(q, directory = file.path(RAW, "GDCdata"))
    saveRDS(se, out)
    log_msg("  SummarizedExperiment: ", nrow(se), " genes x ", ncol(se), " samples")
    record(out, "TCGA-LUAD STAR counts (SummarizedExperiment)")
  }
})

# =============================================================
# 2. Clinical covariates
#
# colData() is the sample description inside the SummarizedExperiment of
# block 1. The covariates are taken from there, NOT from GDCquery_clinic():
# the GDC API now returns multiple diagnosis rows per patient and
# GDCquery_clinic() errors on the length mismatch
# ("Supplied 585 items to be assigned to 943 items"). colData already
# carries the covariates this project uses (age, sex, stage, smoking), plus
# fields from the TCGA marker paper on LUAD (the columns that begin with
# paper_), and its rows are the expression samples themselves, so nothing
# has to be matched.
#
# The survival endpoints still come only from TCGA-CDR (block 3), never
# from here. The file keeps vital_status and the follow-up days for
# reference, but no later step uses them as the outcome.
# =============================================================
run_block("clinical", {
  log_msg("=== TCGA-LUAD clinical covariates (from colData) ===")
  out <- file.path(RAW, "tcga_luad_clinical.rds")
  if (file.exists(out)) {
    log_msg("  already present - skipping")
  } else {
    se_path <- file.path(RAW, "tcga_luad_rnaseq_se.rds")
    if (!file.exists(se_path)) stop("RNA-seq not downloaded yet; run the 'tcga' block first")
    suppressPackageStartupMessages(library(SummarizedExperiment))
    se   <- readRDS(se_path)
    clin <- as.data.frame(colData(se))
    rm(se); invisible(gc())

    # The columns to keep, if they are present. Columns that are not found
    # are named in the log; they do not stop the block.
    keep <- c("barcode", "patient", "bcr_patient_barcode", "sample_type",
              "age_at_diagnosis", "age_at_index", "sex_at_birth", "race", "ethnicity",
              "ajcc_pathologic_stage", "ajcc_pathologic_t", "ajcc_pathologic_n",
              "ajcc_pathologic_m", "pack_years_smoked", "tobacco_smoking_status",
              "tobacco_smoking_onset_year", "tobacco_smoking_quit_year",
              "prior_treatment", "prior_malignancy", "vital_status",
              "days_to_last_follow_up", "days_to_death", "primary_diagnosis",
              "paper_Sex", "paper_Smoking.Status", "paper_Tumor.stage",
              "paper_expression_subtype", "paper_Purity.ABSOLUTE.calls")
    present <- intersect(keep, names(clin))
    missing <- setdiff(keep, names(clin))
    clin <- clin[, present, drop = FALSE]

    log_msg("  ", nrow(clin), " samples x ", length(present), " fields retained")
    if (length(missing)) log_msg("  (not in colData: ", paste(missing, collapse = ", "), ")")
    saveRDS(clin, out)
    record(out, "TCGA-LUAD clinical covariates from colData (NOT endpoints)")
  }
})

# =============================================================
# 3. TCGA-CDR -- the ONLY endpoint source
#
# The TCGA Clinical Data Resource (Liu et al. 2018) is one table with
# curated survival endpoints for every TCGA patient: among them overall
# survival (OS) and the progression-free interval (PFI). Taking the outcome
# from this one curated table means that every patient's endpoint follows
# the same definition.
# =============================================================
run_block("cdr", {
  log_msg("=== TCGA-CDR (survival endpoints) ===")
  fetch_url("tcga_cdr",
            file.path(RAW, "TCGA-CDR-SupplementalTableS1.xlsx"),
            "TCGA-CDR harmonised clinical endpoints (Liu et al. 2018)",
            paste0("Liu et al. (2018) Cell 173(2):400-416, Supplemental Table S1 ",
                   "(https://doi.org/10.1016/j.cell.2018.02.052) - also mirrored on ",
                   "the GDC PanCanAtlas publications page."))
})

# =============================================================
# 4. MC3 consensus somatic mutations
#
# MC3 (Ellrott et al. 2018) is one set of somatic mutation calls for all of
# TCGA. A somatic mutation is one found in the tumour and not in the
# patient's normal tissue. The file is a MAF file (Mutation Annotation
# Format): a tab-separated table with one row per mutation found in one
# tumour. It covers every cancer type; prep/04_preprocess_mutation.py
# keeps the LUAD patients.
# =============================================================
run_block("mc3", {
  log_msg("=== MC3 consensus MAF ===")
  fetch_url("mc3_maf",
            file.path(RAW, "mc3.v0.2.8.PUBLIC.maf.gz"),
            "MC3 pan-cancer consensus somatic mutations (Ellrott et al. 2018)",
            paste0("GDC PanCanAtlas publication page (Ellrott et al. 2018, ",
                   "Cell Systems 6(3):271-281). Large file (~1 GB) - filter to ",
                   "TCGA-LUAD immediately."))
})

# =============================================================
# 5. External GEO cohorts -- each fetched independently
#
# GEO is the Gene Expression Omnibus, a public archive of expression
# studies. Each of the three series in config/config.yaml is downloaded as
# its "series matrix" (the expression table with the sample descriptions)
# together with the description of its microarray platform (getGPL = TRUE),
# which is needed later to turn probe identifiers into gene symbols. The
# result is saved as an ExpressionSet, the R object for such data.
# Each series has its own tryCatch, so one series that fails does not stop
# the other two. If GEO returns more than one matrix for a series, the whole
# list is saved as it is and only the number of matrices is logged.
# =============================================================
run_block("geo", {
  log_msg("=== GEO external cohorts ===")
  suppressPackageStartupMessages(library(GEOquery))
  options(timeout = 3600)
  dir.create(file.path(RAW, "GEO"), showWarnings = FALSE)

  for (g in cfg$data$geo) {
    out <- file.path(RAW, sprintf("%s_eset.rds", g$id))
    if (file.exists(out)) { log_msg("  ", g$id, " already present - skipping"); next }
    log_msg("  fetching ", g$id, " (", g$platform, ", role: ", g$role, ") ...")
    tryCatch({
      gse  <- GEOquery::getGEO(g$id, GSEMatrix = TRUE, getGPL = TRUE,
                               destdir = file.path(RAW, "GEO"))
      eset <- if (length(gse) == 1) gse[[1]] else gse
      saveRDS(eset, out)
      if (length(gse) == 1) {
        log_msg("    ", nrow(Biobase::exprs(eset)), " features x ",
                ncol(Biobase::exprs(eset)), " samples")
      } else {
        log_msg("    ", length(gse), " series matrices")
      }
      record(out, sprintf("GEO %s (%s) - %s", g$id, g$platform, g$role))
    }, error = function(e) {
      failures <<- c(failures, g$id)
      log_msg("    !! FAILED: ", g$id)
      message("       ", conditionMessage(e))
    })
  }
})

# =============================================================
# Summary: list the files now in data/raw/ with their sizes, name the
# blocks that failed and print the command that runs just those again.
# The last line printed, "Next: make overlap", names a target of the
# earlier project that does not exist here. The next steps here are
# `make download-methylation` and then `make prep`.
log_msg("=== summary ===")
files <- list.files(RAW, pattern = "\\.(rds|xlsx|gz)$", full.names = TRUE)
if (length(files)) {
  for (f in files) cat(sprintf("  %-45s %8.1f MB\n", basename(f), file.info(f)$size / 1e6))
} else cat("  no data files yet\n")

if (length(failures)) {
  cat("\nFAILED BLOCKS: ", paste(unique(failures), collapse = ", "), "\n")
  cat("Re-run just those, e.g.:  Rscript prep/01_download.R ",
      paste(unique(failures), collapse = " "), "\n", sep = "")
} else {
  cat("\nAll requested blocks completed.\n")
}
cat("\nManifest: ", file.path(RAW, "MANIFEST.md"), "\n")
cat("Next: make overlap\n")
if (length(failures)) quit(status = 1)
