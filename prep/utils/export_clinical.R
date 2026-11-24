#!/usr/bin/env Rscript
# =============================================================
# utils/export_clinical.R
#
# Exports the two phenotype tables Python needs, in patient-barcode space:
#
#   data/interim/clinical_raw.parquet   covariates, one row per patient
#   data/interim/endpoints.parquet      PFI (primary) and OS (sensitivity)
#
# No imputation, no scaling, no dropping of incomplete rows: those are
# fold-internal decisions and belong in 05_preprocess_clinical.py /
# 06_nested_cv.py. This script only harmonises sources and encodings.
#
# Covariate sourcing is deliberately split:
#   stage / age / sex  <- TCGA-CDR, falling back to GDC colData
#   smoking            <- GDC colData (TCGA-CDR carries no smoking fields)
# GDC colData records no pathologic stage for 121 of the 517 primary
# tumours (23%), against 8 of 522 in TCGA-CDR. Taking stage from colData
# would throw away the single strongest clinical prognostic factor for a
# quarter of the cohort, so the curated table wins wherever both exist.
# Every field carries a *_source column so the mix is auditable.
# =============================================================

suppressPackageStartupMessages({
  library(yaml); library(arrow); library(data.table); library(readxl)
})

cfg     <- yaml::read_yaml("config/config.yaml")
RAW     <- cfg$paths$raw
INTERIM <- cfg$paths$interim
dir.create(INTERIM, recursive = TRUE, showWarnings = FALSE)

log_msg <- function(...) cat(format(Sys.time(), "[%H:%M:%S] "), ..., "\n", sep = "")

patient_of     <- function(bc) substr(bc, 1, 12)
sample_type_of <- function(bc) substr(bc, 14, 15)
vial_of        <- function(bc) substr(bc, 16, 16)
plate_of       <- function(bc) substr(bc, 22, 25)

# ---------------------------------------------------------------
# Encodings. Anything not matched maps to NA and is carried as
# missing rather than guessed -- 05 adds explicit indicators.
# ---------------------------------------------------------------
STAGE_ORDINAL <- c(
  "Stage I" = 1, "Stage IA" = 1, "Stage IB" = 1,
  "Stage II" = 2, "Stage IIA" = 2, "Stage IIB" = 2,
  "Stage III" = 3, "Stage IIIA" = 3, "Stage IIIB" = 3,
  "Stage IV" = 4
)

# Ordinal in exposure recency: never < remote former < recent former < current.
# "Duration Not Specified" is a genuine former smoker with an unplaceable
# rank, so it is NA on the ordinal but 1 on ever_smoker -- discarding the
# ordinal is not the same as discarding the patient.
SMOKING_ORDINAL <- c(
  "Lifelong Non-Smoker" = 0,
  "Current Reformed Smoker for > 15 yrs" = 1,
  "Current Reformed Smoker for < or = 15 yrs" = 2,
  "Current Smoker" = 3
)
SMOKING_EVER <- c(
  "Lifelong Non-Smoker" = 0,
  "Current Reformed Smoker for > 15 yrs" = 1,
  "Current Reformed Smoker for < or = 15 yrs" = 1,
  "Current Reformed Smoker, Duration Not Specified" = 1,
  "Current Smoker" = 1
)

# TCGA sentinels for "no value", used across both sources.
blank_to_na <- function(x) {
  x <- as.character(x)
  x[x %in% c("[Not Available]", "[Not Applicable]", "[Unknown]", "[Discrepancy]",
             "Not Reported", "Unknown", "NA", "")] <- NA_character_
  x
}

# ---------------------------------------------------------------
# GDC colData -- smoking, and the fallback for age / sex / stage
# ---------------------------------------------------------------
f_clin <- file.path(RAW, "tcga_luad_clinical.rds")
if (!file.exists(f_clin)) stop("missing ", f_clin, "\n  run `make download` first")

log_msg("=== GDC colData ===")
clin <- as.data.table(readRDS(f_clin))
log_msg("  rows as downloaded: ", nrow(clin))

clin <- clin[sample_type_of(barcode) %in% cfg$data$tcga$sample_types]
log_msg("  primary solid tumour (type 01): ", nrow(clin))

# Same aliquot rule as export_barcodes.R: prefer vial A, then lowest plate.
# Both scripts must land on the same patient set or the availability matrix
# and the design matrix describe different cohorts.
clin[, `:=`(patient_barcode = patient_of(barcode),
            vial = vial_of(barcode), plate = plate_of(barcode))]
setorder(clin, patient_barcode, vial, plate)
clin <- clin[, .SD[1], by = patient_barcode]
log_msg("  one aliquot per patient: ", nrow(clin))

gdc <- clin[, .(
  patient_barcode,
  gdc_age       = suppressWarnings(as.numeric(age_at_index)),
  gdc_sex       = tolower(blank_to_na(sex_at_birth)),
  gdc_stage     = blank_to_na(ajcc_pathologic_stage),
  smoking_raw   = blank_to_na(tobacco_smoking_status),
  pack_years    = suppressWarnings(as.numeric(pack_years_smoked))
)]

# paper_Tumor.stage is the TCGA marker-paper stage call; it fills some of
# colData's gaps and is only ever used when colData itself is empty.
if ("paper_Tumor.stage" %in% names(clin)) {
  paper_stage <- blank_to_na(clin$paper_Tumor.stage)
  n_rescued <- sum(is.na(gdc$gdc_stage) & !is.na(paper_stage))
  gdc[, gdc_stage := fifelse(is.na(gdc_stage), paper_stage, gdc_stage)]
  log_msg("  stage: ", n_rescued, " gaps filled from paper_Tumor.stage")
}

# ---------------------------------------------------------------
# TCGA-CDR -- endpoints, and the primary source for age / sex / stage
# ---------------------------------------------------------------
f_cdr <- file.path(RAW, "TCGA-CDR-SupplementalTableS1.xlsx")
if (!file.exists(f_cdr)) stop("missing ", f_cdr, "\n  run `make download` first")

log_msg("=== TCGA-CDR ===")
cdr <- as.data.table(suppressWarnings(readxl::read_excel(f_cdr, sheet = 1)))
# Column punctuation varies between distributions of this table
# (PFI.time / PFI_time / pfi time); normalise before looking anything up.
setnames(cdr, tolower(gsub("[^A-Za-z0-9]+", "_", names(cdr))))
need <- c("bcr_patient_barcode", "type", "pfi", "pfi_time", "os", "os_time",
          "age_at_initial_pathologic_diagnosis", "gender",
          "ajcc_pathologic_tumor_stage")
absent <- setdiff(need, names(cdr))
if (length(absent))
  stop("TCGA-CDR: expected columns not found: ", paste(absent, collapse = ", "),
       "\n  present: ", paste(head(names(cdr), 40), collapse = ", "))

cdr <- cdr[type == "LUAD"]
log_msg("  LUAD patients: ", nrow(cdr))
cdr[, patient_barcode := patient_of(as.character(bcr_patient_barcode))]

if ("redaction" %in% names(cdr)) {
  n0 <- nrow(cdr)
  cdr <- cdr[is.na(redaction) | redaction != "Redacted"]
  log_msg("  after dropping redacted: ", nrow(cdr), " (dropped ", n0 - nrow(cdr), ")")
}
if (anyDuplicated(cdr$patient_barcode))
  stop("TCGA-CDR: duplicate patient barcodes after redaction filter")

cdr_cov <- cdr[, .(
  patient_barcode,
  cdr_age   = suppressWarnings(as.numeric(age_at_initial_pathologic_diagnosis)),
  cdr_sex   = tolower(blank_to_na(gender)),
  cdr_stage = blank_to_na(ajcc_pathologic_tumor_stage)
)]

# ---------------------------------------------------------------
# Covariates: CDR first, colData second, source recorded per field
# ---------------------------------------------------------------
cov <- merge(cdr_cov, gdc, by = "patient_barcode", all = TRUE)

coalesce_src <- function(primary, fallback, name) {
  out <- fifelse(is.na(primary), fallback, primary)
  src <- fifelse(!is.na(primary), "cdr",
                 fifelse(!is.na(fallback), "gdc", NA_character_))
  log_msg("  ", name, ": ", sum(src == "cdr", na.rm = TRUE), " from CDR, ",
          sum(src == "gdc", na.rm = TRUE), " from colData, ",
          sum(is.na(out)), " missing")
  list(value = out, source = src)
}

log_msg("=== covariate sourcing ===")
age   <- coalesce_src(cov$cdr_age, cov$gdc_age, "age")
sex   <- coalesce_src(cov$cdr_sex, cov$gdc_sex, "sex")
stage <- coalesce_src(cov$cdr_stage, cov$gdc_stage, "stage")

clinical <- data.table(
  patient_barcode = cov$patient_barcode,
  age             = age$value,
  age_source      = age$source,
  sex_male        = fifelse(is.na(sex$value), NA_real_,
                            as.numeric(sex$value == "male")),
  sex_source      = sex$source,
  stage_raw       = stage$value,
  stage_ordinal   = unname(STAGE_ORDINAL[stage$value]),
  stage_source    = stage$source,
  smoking_raw     = cov$smoking_raw,
  smoking_ordinal = unname(SMOKING_ORDINAL[cov$smoking_raw]),
  ever_smoker     = unname(SMOKING_EVER[cov$smoking_raw]),
  pack_years      = cov$pack_years
)
setorder(clinical, patient_barcode)

# A stage string present but unmapped means a level this script has never
# seen; silently emitting NA would quietly drop the strongest covariate.
unmapped <- setdiff(unique(na.omit(clinical$stage_raw)), names(STAGE_ORDINAL))
if (length(unmapped))
  log_msg("  !! unmapped stage levels -> NA: ", paste(unmapped, collapse = " | "))
unmapped_smk <- setdiff(unique(na.omit(clinical$smoking_raw)), names(SMOKING_EVER))
if (length(unmapped_smk))
  log_msg("  !! unmapped smoking levels -> NA: ", paste(unmapped_smk, collapse = " | "))

# ---------------------------------------------------------------
# Endpoints. Times stay in days and uncensored here; the 60-month
# administrative censor in config is applied at load time in Python so
# it is visible next to the model that uses it.
# ---------------------------------------------------------------
endpoints <- cdr[, .(
  patient_barcode,
  pfi_event = suppressWarnings(as.integer(pfi)),
  pfi_time  = suppressWarnings(as.numeric(pfi_time)),
  os_event  = suppressWarnings(as.integer(os)),
  os_time   = suppressWarnings(as.numeric(os_time))
)]
n0 <- nrow(endpoints)
endpoints <- endpoints[!is.na(pfi_event) & !is.na(pfi_time) & pfi_time > 0]
setorder(endpoints, patient_barcode)

log_msg("=== endpoints ===")
log_msg("  usable PFI (time > 0): ", nrow(endpoints), " (dropped ", n0 - nrow(endpoints), ")")
log_msg("  PFI events: ", sum(endpoints$pfi_event == 1), " (",
        round(100 * mean(endpoints$pfi_event == 1), 1), "%)")
log_msg("  OS events:  ", sum(endpoints$os_event == 1, na.rm = TRUE), " (",
        round(100 * mean(endpoints$os_event == 1, na.rm = TRUE), 1), "%)")

# ---------------------------------------------------------------
out_clin <- file.path(INTERIM, "clinical_raw.parquet")
out_endp <- file.path(INTERIM, "endpoints.parquet")
arrow::write_parquet(clinical,  out_clin)
arrow::write_parquet(endpoints, out_endp)

log_msg("=== written ===")
log_msg("  ", out_clin, "  (", nrow(clinical), " patients, ", ncol(clinical), " cols)")
log_msg("  ", out_endp, "  (", nrow(endpoints), " patients)")
log_msg("Next: Rscript prep/03_preprocess_rna.R")
