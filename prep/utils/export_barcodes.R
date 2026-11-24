#!/usr/bin/env Rscript
# =============================================================
# export_barcodes.R -- count the patients each data source has, step by step
#
# What it is for
#   Goes through the raw downloads (RNA-seq, clinical covariates, the
#   TCGA-CDR endpoint table, the MC3 mutation file), applies the sample
#   selection rules, and writes down how many samples or patients remain
#   after each rule, and which patients each source ends up with.
#   This is a record for the reader: it shows where patients are lost.
#   The analysis scripts do not read the two files written here, and the
#   methylation block is not part of this count.
#
# Where it sits
#   Step 2 of 9 of `make prep`, after `make download`.
#   Run before prep/utils/export_clinical.R.
#
# Reads    data/raw/tcga_luad_rnaseq_se.rds
#          data/raw/tcga_luad_clinical.rds
#          data/raw/TCGA-CDR-SupplementalTableS1.xlsx
#          data/raw/mc3.v0.2.8.PUBLIC.maf.gz
#          config/config.yaml  (sample type, mutation filter, variant types)
# Writes   data/interim/barcodes.parquet    long table: block, patient_barcode
#          data/interim/consort_counts.csv  step-by-step selection counts
#          ("CONSORT" is the standard for the flow diagram of a clinical
#          trial report: how many were assessed, excluded and analysed.)
#
# How to run (from the project folder)
#   Rscript prep/utils/export_barcodes.R
#
# The downloads are R objects (.rds) that Python cannot read; the parquet
# file written here can be read by both.
#
# A source that has not been downloaded yet is skipped with a message and
# is simply absent from the output. The script stops only if no source at
# all is present.
# =============================================================

# yaml reads the configuration, arrow writes parquet files, data.table
# reads and handles tables. readxl is called once below without being
# loaded here (readxl::read_excel), so it must be installed.
suppressPackageStartupMessages({
  library(yaml); library(arrow); library(data.table)
})

# Folders from the configuration: raw data and intermediate files.
cfg     <- yaml::read_yaml("config/config.yaml")
RAW     <- cfg$paths$raw
INTERIM <- cfg$paths$interim
dir.create(INTERIM, recursive = TRUE, showWarnings = FALSE)

# Print a message with the time of day in front of it.
log_msg <- function(...) cat(format(Sys.time(), "[%H:%M:%S] "), ..., "\n", sep = "")

# Every TCGA sample has a barcode, an identifier whose parts have fixed
# positions. The four helpers below cut the parts out.
#
# TCGA barcode: TCGA-XX-XXXX-01A-11R-XXXX-07
#               |--patient--| |ss||v|  |plate|
#
#   patient  characters 1-12
#   ss       characters 14-15, the sample type: 01 is a primary solid
#            tumour, 11 is normal tissue
#   v        character 16, the vial: a letter, A for the first, when a
#            sample was divided into several vials
#   plate    characters 22-25, the plate the material was processed on
#
# An aliquot is the portion of a sample's material that was actually
# measured. The full barcode names one aliquot; a patient can have several.
patient_of <- function(bc) substr(bc, 1, 12)
sample_type_of <- function(bc) substr(bc, 14, 15)
vial_of <- function(bc) substr(bc, 16, 16)
plate_of <- function(bc) substr(bc, 22, 25)

# `consort` collects the counts: one row per selection step. add_step() adds
# a row and prints it.
consort <- data.table(step = character(), n = integer(), detail = character())
add_step <- function(step, n, detail = "") {
  consort <<- rbind(consort, data.table(step = step, n = as.integer(n), detail = detail))
  log_msg("  ", step, ": ", n, if (nzchar(detail)) paste0("  (", detail, ")") else "")
}

# `blocks` will hold, for each source, the patients it has after selection.
# `luad_roster` will hold every LUAD patient named in TCGA-CDR; it is used
# further down to pick the LUAD rows out of the mutation file.
blocks <- list()
luad_roster <- NULL

# ---------------------------------------------------------------
# RNA-seq: from all downloaded aliquots to one tumour sample per patient.
#
# Two rules, the same as in prep/03_preprocess_rna.R:
#   1. keep primary solid tumours (sample type 01);
#   2. keep one aliquot per patient. Some patients were sequenced more than
#      once; keeping all aliquots would count them twice.
# ---------------------------------------------------------------
f <- file.path(RAW, "tcga_luad_rnaseq_se.rds")
if (file.exists(f)) {
  log_msg("=== RNA-seq ===")
  se <- readRDS(f)
  bc <- colnames(se)
  add_step("RNA: aliquots downloaded", length(bc))

  keep <- sample_type_of(bc) %in% cfg$data$tcga$sample_types
  add_step("RNA: primary solid tumour (type 01)", sum(keep),
           paste("dropped", sum(!keep), "non-01 samples"))
  bc <- bc[keep]

  # One aliquot per patient: prefer vial A, then the lowest plate. The
  # table is sorted by vial and plate, and `.SD[1]` takes the first row of
  # each patient. The rule is fixed and never looks at the data.
  dt <- data.table(barcode = bc, patient = patient_of(bc),
                   vial = vial_of(bc), plate = plate_of(bc))
  n_dup_patients <- sum(duplicated(dt$patient))
  setorder(dt, patient, vial, plate)
  dt <- dt[, .SD[1], by = patient]
  add_step("RNA: one aliquot per patient", nrow(dt),
           paste("resolved", n_dup_patients, "duplicate aliquots"))

  blocks$rna <- dt$patient
  rm(se); invisible(gc())
} else {
  log_msg("RNA-seq not downloaded yet - skipping")
}

# ---------------------------------------------------------------
# Clinical covariates from the GDC (age, sex, stage, smoking). The survival
# outcome is not taken from here.
# ---------------------------------------------------------------
f <- file.path(RAW, "tcga_luad_clinical.rds")
if (file.exists(f)) {
  log_msg("=== Clinical covariates ===")
  clin <- readRDS(f)
  # The column "bcr_patient_barcode" of this file is SAMPLE-level
  # (TCGA-XX-XXXX-01A), not patient-level, despite its name. Every source
  # is therefore cut to the 12-character patient barcode. patient_of()
  # changes nothing if the identifier already has 12 characters.
  idcol <- if ("patient" %in% names(clin)) "patient" else "bcr_patient_barcode"
  ids <- unique(patient_of(as.character(clin[[idcol]])))
  add_step("Clinical: patients with covariates", length(ids))
  blocks$clin <- ids
} else {
  log_msg("Clinical covariates not extracted yet - skipping")
}

# ---------------------------------------------------------------
# TCGA-CDR: the ONLY source of the survival endpoints.
#
# The primary endpoint is the progression-free interval (PFI): the time
# from diagnosis until the disease progresses or comes back, or the patient
# dies with tumour. The table has two columns for it: PFI is 1 if that
# event was seen and 0 if it was not, and PFI.time is the number of days
# until the event or until the patient was last seen. A patient with PFI 0
# is "censored": all that is known is that the event had not happened by
# the last contact.
# ---------------------------------------------------------------
f <- file.path(RAW, "TCGA-CDR-SupplementalTableS1.xlsx")
if (file.exists(f)) {
  log_msg("=== TCGA-CDR ===")
  cdr <- as.data.table(readxl::read_excel(f, sheet = 1))
  # Column names vary in punctuation between distributions of this table
  # (PFI.time / PFI_time / pfi time). All names are changed to lower case
  # with underscores first. col() then looks a column up by that name and
  # stops with a clear message if a required one is missing.
  setnames(cdr, tolower(gsub("[^A-Za-z0-9]+", "_", names(cdr))))
  col <- function(want, required = TRUE) {
    hit <- names(cdr)[names(cdr) == want]
    if (!length(hit) && required)
      stop("TCGA-CDR: expected column '", want, "' not found. Present: ",
           paste(head(names(cdr), 40), collapse = ", "))
    if (length(hit)) hit[1] else NA_character_
  }
  c_type <- col("type"); c_bc <- col("bcr_patient_barcode")
  c_pfi  <- col("pfi");  c_pfit <- col("pfi_time")
  c_red  <- col("redaction", required = FALSE)

  # The table covers all of TCGA; keep lung adenocarcinoma.
  cdr <- cdr[get(c_type) == "LUAD"]
  add_step("CDR: LUAD patients", nrow(cdr))
  # The full list of LUAD patients, whatever data they have. It is used
  # below to pick the LUAD rows out of the mutation file, which covers
  # every cancer type, without making that choice depend on RNA-seq.
  luad_roster <- patient_of(as.character(cdr[[c_bc]]))

  # The table marks patients whose data were withdrawn as "Redacted".
  # They are removed, if the column exists.
  if (!is.na(c_red)) {
    n0 <- nrow(cdr)
    cdr <- cdr[is.na(get(c_red)) | get(c_red) != "Redacted"]
    add_step("CDR: after dropping redacted", nrow(cdr), paste("dropped", n0 - nrow(cdr)))
  }
  # A usable endpoint needs a time above zero and a known event status.
  # Text that is not a number becomes missing (NA) here; the warnings R
  # would give for that are switched off because it is expected.
  pfi <- suppressWarnings(as.numeric(cdr[[c_pfit]]))
  ev  <- suppressWarnings(as.numeric(cdr[[c_pfi]]))
  ok  <- !is.na(pfi) & pfi > 0 & !is.na(ev)
  add_step("CDR: usable PFI (time > 0)", sum(ok), paste("dropped", sum(!ok)))
  cdr <- cdr[ok]
  add_step("CDR: PFI events", sum(suppressWarnings(as.numeric(cdr[[c_pfi]])) == 1),
           paste0(round(100 * mean(suppressWarnings(as.numeric(cdr[[c_pfi]])) == 1), 1), "% event rate"))

  blocks$cdr <- patient_of(as.character(cdr[[c_bc]]))
} else {
  log_msg("TCGA-CDR not present - skipping (set data.urls.tcga_cdr in config.yaml)")
}

# ---------------------------------------------------------------
# MC3 mutations.
#
# The file is a MAF file (Mutation Annotation Format): a table with one
# row per mutation found in one tumour, for every TCGA cancer type. Only
# three columns are read, because the file is large:
#   Tumor_Sample_Barcode    which tumour the mutation was found in
#   FILTER                  quality flags; PASS means none was raised
#   Variant_Classification  the kind of change (missense, silent, ...)
# ---------------------------------------------------------------
f <- file.path(RAW, "mc3.v0.2.8.PUBLIC.maf.gz")
if (file.exists(f)) {
  log_msg("=== MC3 MAF (large file - reading selected columns only) ===")
  maf <- fread(f, select = c("Tumor_Sample_Barcode", "FILTER", "Variant_Classification"),
               showProgress = FALSE)
  add_step("MC3: variants (pan-cancer)", nrow(maf))

  maf[, patient := patient_of(Tumor_Sample_Barcode)]
  # Keep the LUAD patients, using the list from TCGA-CDR where it is
  # available. The RNA-seq patients are NOT used for this: the mutation
  # block would then contain only patients who also have RNA-seq, and the
  # count of patients with mutation data would describe that filter rather
  # than the data. If TCGA-CDR is absent, the clinical patients are used;
  # if both are absent, the counts stay pan-cancer and say so.
  if (!is.null(luad_roster)) {
    maf <- maf[patient %in% luad_roster]
    add_step("MC3: variants in LUAD (CDR roster)", nrow(maf))
  } else if (!is.null(blocks$clin)) {
    maf <- maf[patient %in% blocks$clin]
    add_step("MC3: variants in LUAD (clinical roster)", nrow(maf),
             "CDR absent - roster from clinical covariates")
  } else {
    add_step("MC3: variants (pan-cancer, unscoped)", nrow(maf),
             "no LUAD roster available - counts are pan-cancer")
  }

  # Keep calls that passed every quality filter (PASS, from the config).
  maf <- maf[FILTER == cfg$data$mutation$filter]
  add_step("MC3: PASS variants", nrow(maf))

  # Keep non-silent mutations: those that change the protein (the kinds
  # listed in the config). A silent mutation changes the DNA but not the
  # protein, and is not counted.
  maf <- maf[Variant_Classification %in% unlist(cfg$data$mutation$nonsilent)]
  add_step("MC3: non-silent variants", nrow(maf))

  blocks$mut <- unique(maf$patient)
  add_step("MC3: patients with >= 1 non-silent variant", length(blocks$mut))
  rm(maf); invisible(gc())
} else {
  log_msg("MC3 MAF not present - skipping (set data.urls.mc3_maf in config.yaml)")
}

# ---------------------------------------------------------------
# Write the two outputs.
#
# `long` has one row per source and patient ("long" format), which is easy
# to count and to compare between sources.
# ---------------------------------------------------------------
if (!length(blocks)) {
  stop("No blocks available. Run: Rscript prep/01_download.R")
}

long <- rbindlist(lapply(names(blocks), function(b) {
  data.table(block = b, patient_barcode = unique(blocks[[b]]))
}))

arrow::write_parquet(long, file.path(INTERIM, "barcodes.parquet"))
fwrite(consort, file.path(INTERIM, "consort_counts.csv"))

log_msg("=== written ===")
log_msg("  ", file.path(INTERIM, "barcodes.parquet"), "  (",
        nrow(long), " rows, ", length(blocks), " blocks)")
log_msg("  ", file.path(INTERIM, "consort_counts.csv"))
log_msg("Next: Rscript prep/utils/build_pathways.R")
