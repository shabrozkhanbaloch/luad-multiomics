#!/usr/bin/env Rscript
# =============================================================
# utils/export_barcodes.R
#
# Extracts the patient barcodes present in each omics block from the
# raw downloads, applies the sample-selection rules, and writes:
#
#   data/interim/barcodes.parquet   long table: block, patient_barcode
#   data/interim/consort_counts.csv step-by-step selection counts
#
# The R objects are .rds; Python cannot read them, so this is the
# handoff. Run before prep/utils/export_clinical.R.
#
# Blocks that have not been downloaded yet are simply absent from the
# output - 02_overlap_matrix.py reports which are missing.
# =============================================================

suppressPackageStartupMessages({
  library(yaml); library(arrow); library(data.table)
})

cfg     <- yaml::read_yaml("config/config.yaml")
RAW     <- cfg$paths$raw
INTERIM <- cfg$paths$interim
dir.create(INTERIM, recursive = TRUE, showWarnings = FALSE)

log_msg <- function(...) cat(format(Sys.time(), "[%H:%M:%S] "), ..., "\n", sep = "")

# TCGA barcode: TCGA-XX-XXXX-01A-11R-XXXX-07
#               |--patient--| |ss||v|  |plate|
patient_of <- function(bc) substr(bc, 1, 12)
sample_type_of <- function(bc) substr(bc, 14, 15)
vial_of <- function(bc) substr(bc, 16, 16)
plate_of <- function(bc) substr(bc, 22, 25)

consort <- data.table(step = character(), n = integer(), detail = character())
add_step <- function(step, n, detail = "") {
  consort <<- rbind(consort, data.table(step = step, n = as.integer(n), detail = detail))
  log_msg("  ", step, ": ", n, if (nzchar(detail)) paste0("  (", detail, ")") else "")
}

blocks <- list()
luad_roster <- NULL

# ---------------------------------------------------------------
# RNA-seq: full CONSORT-style selection
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

  # one aliquot per patient: prefer vial A, then lowest plate
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
# GDC clinical (covariates only)
# ---------------------------------------------------------------
f <- file.path(RAW, "tcga_luad_clinical.rds")
if (file.exists(f)) {
  log_msg("=== Clinical covariates ===")
  clin <- readRDS(f)
  # colData's "bcr_patient_barcode" is SAMPLE-level (TCGA-XX-XXXX-01A), not
  # patient-level, despite the name. Normalise every block to the 12-char
  # patient barcode; patient_of() is a no-op on an already-12-char id.
  idcol <- if ("patient" %in% names(clin)) "patient" else "bcr_patient_barcode"
  ids <- unique(patient_of(as.character(clin[[idcol]])))
  add_step("Clinical: patients with covariates", length(ids))
  blocks$clin <- ids
} else {
  log_msg("Clinical covariates not extracted yet - skipping")
}

# ---------------------------------------------------------------
# TCGA-CDR: the ONLY endpoint source
# ---------------------------------------------------------------
f <- file.path(RAW, "TCGA-CDR-SupplementalTableS1.xlsx")
if (file.exists(f)) {
  log_msg("=== TCGA-CDR ===")
  cdr <- as.data.table(readxl::read_excel(f, sheet = 1))
  # Column names vary in punctuation between distributions of this table
  # (PFI.time / PFI_time / pfi time). Normalise, then look up by canonical name.
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

  cdr <- cdr[get(c_type) == "LUAD"]
  add_step("CDR: LUAD patients", nrow(cdr))
  # Full LUAD roster, independent of any omics availability. Used to scope
  # the pan-cancer MC3 MAF to LUAD without conditioning on RNA.
  luad_roster <- patient_of(as.character(cdr[[c_bc]]))

  if (!is.na(c_red)) {
    n0 <- nrow(cdr)
    cdr <- cdr[is.na(get(c_red)) | get(c_red) != "Redacted"]
    add_step("CDR: after dropping redacted", nrow(cdr), paste("dropped", n0 - nrow(cdr)))
  }
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
# MC3 mutations
# ---------------------------------------------------------------
f <- file.path(RAW, "mc3.v0.2.8.PUBLIC.maf.gz")
if (file.exists(f)) {
  log_msg("=== MC3 MAF (large file - reading selected columns only) ===")
  maf <- fread(f, select = c("Tumor_Sample_Barcode", "FILTER", "Variant_Classification"),
               showProgress = FALSE)
  add_step("MC3: variants (pan-cancer)", nrow(maf))

  maf[, patient := patient_of(Tumor_Sample_Barcode)]
  # Scope to LUAD using the CDR roster where available. Do NOT scope by the
  # RNA cohort: that would make the mutation block a subset of RNA by
  # construction, and the availability matrix would then be measuring the
  # filter rather than the data.
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

  maf <- maf[FILTER == cfg$data$mutation$filter]
  add_step("MC3: PASS variants", nrow(maf))

  maf <- maf[Variant_Classification %in% unlist(cfg$data$mutation$nonsilent)]
  add_step("MC3: non-silent variants", nrow(maf))

  blocks$mut <- unique(maf$patient)
  add_step("MC3: patients with >= 1 non-silent variant", length(blocks$mut))
  rm(maf); invisible(gc())
} else {
  log_msg("MC3 MAF not present - skipping (set data.urls.mc3_maf in config.yaml)")
}

# ---------------------------------------------------------------
# Write outputs
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
log_msg("Next: Rscript prep/utils/export_clinical.R")
