#!/usr/bin/env Rscript
# =============================================================
# export_clinical.R -- one clinical record and one outcome record per patient
#
# What it is for
#   Brings together the two sources of patient information and writes two
#   tables, one row per patient, named by the 12-character patient barcode:
#   the clinical covariates (age, sex, stage, smoking) and the survival
#   endpoints.
#
# Where it sits
#   Step 6 of 9 of `make prep`, after `make download`.
#   prep/04_preprocess_mutation.py and prep/05_preprocess_clinical.py read
#   the covariate table written here, so this script must have run before
#   either of them.
#
# Reads    data/raw/tcga_luad_clinical.rds             covariates from the GDC
#          data/raw/TCGA-CDR-SupplementalTableS1.xlsx  endpoints and covariates
#          config/config.yaml                          (sample type)
# Writes   data/interim/clinical_raw.parquet   covariates, one row per patient
#          data/interim/endpoints.parquet      PFI (primary) and OS (sensitivity);
#                                              the analysis reads this file directly
#
# How to run (from the project folder)
#   Rscript prep/utils/export_clinical.R
#
# What is NOT done here
#   No filling-in of missing values (imputation), no scaling, no dropping
#   of patients with incomplete records. Those choices depend on the data,
#   so they are made on the training patients inside each cross-validation
#   fold (src/luad/transforms.py). This script only brings the two sources
#   together and turns text into numbers.
#
# Where each covariate comes from
#   stage / age / sex  <- TCGA-CDR, falling back to the GDC file (colData)
#   smoking            <- the GDC file (TCGA-CDR carries no smoking fields)
# GDC colData records no pathologic stage for 121 of the 517 primary
# tumours (23%), against 8 of 522 in TCGA-CDR. Taking stage from colData
# would throw away the single strongest clinical prognostic factor for a
# quarter of the cohort, so the curated table wins wherever both exist.
# Every field carries a *_source column ("cdr" or "gdc"), so that it can
# be checked where each value came from.
# =============================================================

# yaml reads the configuration, arrow writes parquet files, data.table
# handles tables, readxl reads the Excel file of TCGA-CDR.
suppressPackageStartupMessages({
  library(yaml); library(arrow); library(data.table); library(readxl)
})

# Folders from the configuration: raw data and intermediate files.
cfg     <- yaml::read_yaml("config/config.yaml")
RAW     <- cfg$paths$raw
INTERIM <- cfg$paths$interim
dir.create(INTERIM, recursive = TRUE, showWarnings = FALSE)

# Print a message with the time of day in front of it.
log_msg <- function(...) cat(format(Sys.time(), "[%H:%M:%S] "), ..., "\n", sep = "")

# Helpers that cut the parts out of a TCGA barcode such as
# TCGA-XX-XXXX-01A-11R-XXXX-07: the patient (characters 1-12), the sample
# type (14-15; 01 is a primary solid tumour), the vial (16) and the plate
# (22-25). The parts are explained in prep/utils/export_barcodes.R.
patient_of     <- function(bc) substr(bc, 1, 12)
sample_type_of <- function(bc) substr(bc, 14, 15)
vial_of        <- function(bc) substr(bc, 16, 16)
plate_of       <- function(bc) substr(bc, 22, 25)

# ---------------------------------------------------------------
# Encodings: how text becomes a number.
#
# A value that is not in a table below becomes missing (NA). It is carried
# as missing rather than guessed; prep/05_preprocess_clinical.py later adds
# a 0/1 column that says "this value was missing".
#
# Stage: the sub-stages are merged (IA and IB both become 1), giving an
# ordered scale from 1 (stage I) to 4 (stage IV).
# ---------------------------------------------------------------
STAGE_ORDINAL <- c(
  "Stage I" = 1, "Stage IA" = 1, "Stage IB" = 1,
  "Stage II" = 2, "Stage IIA" = 2, "Stage IIB" = 2,
  "Stage III" = 3, "Stage IIIA" = 3, "Stage IIIB" = 3,
  "Stage IV" = 4
)

# Smoking, as two variables.
# SMOKING_ORDINAL orders patients by how recent the exposure is:
# never (0) < stopped more than 15 years ago (1) < stopped within 15 years
# (2) < current smoker (3).
# SMOKING_EVER is 0 for lifelong non-smokers and 1 for everyone else.
# "Duration Not Specified" is a real former smoker whose place on the
# ordered scale is unknown, so that patient is NA on the ordinal but 1 on
# ever_smoker. Losing the rank is not a reason to lose the patient.
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

# TCGA writes "no value" in several ways ("[Not Available]", "Unknown",
# an empty string, ...). blank_to_na() turns all of them into NA, for both
# sources.
blank_to_na <- function(x) {
  x <- as.character(x)
  x[x %in% c("[Not Available]", "[Not Applicable]", "[Unknown]", "[Discrepancy]",
             "Not Reported", "Unknown", "NA", "")] <- NA_character_
  x
}

# ---------------------------------------------------------------
# The GDC file (colData) -- smoking, and the fallback for age / sex / stage.
#
# The file has one row per sequenced aliquot (see export_barcodes.R for
# the word), so it is first cut to one row per patient.
# ---------------------------------------------------------------
f_clin <- file.path(RAW, "tcga_luad_clinical.rds")
if (!file.exists(f_clin)) stop("missing ", f_clin, "\n  run `make download` first")

log_msg("=== GDC colData ===")
clin <- as.data.table(readRDS(f_clin))
log_msg("  rows as downloaded: ", nrow(clin))

clin <- clin[sample_type_of(barcode) %in% cfg$data$tcga$sample_types]
log_msg("  primary solid tumour (type 01): ", nrow(clin))

# Same aliquot rule as export_barcodes.R: prefer vial A, then lowest plate.
# A patient with several aliquots would otherwise appear in several rows.
# Both scripts must arrive at the same patients, or the patient counts and
# the covariate table would describe different groups.
clin[, `:=`(patient_barcode = patient_of(barcode),
            vial = vial_of(barcode), plate = plate_of(barcode))]
setorder(clin, patient_barcode, vial, plate)
clin <- clin[, .SD[1], by = patient_barcode]
log_msg("  one aliquot per patient: ", nrow(clin))

# Take the five fields needed, clean the text and turn numbers into
# numbers. Pack-years measure lifetime smoking: packs per day multiplied
# by years smoked.
gdc <- clin[, .(
  patient_barcode,
  gdc_age       = suppressWarnings(as.numeric(age_at_index)),
  gdc_sex       = tolower(blank_to_na(sex_at_birth)),
  gdc_stage     = blank_to_na(ajcc_pathologic_stage),
  smoking_raw   = blank_to_na(tobacco_smoking_status),
  pack_years    = suppressWarnings(as.numeric(pack_years_smoked))
)]

# paper_Tumor.stage is the stage given in the TCGA marker paper on LUAD. It
# fills some of colData's gaps, and is only ever used when colData itself
# has no stage. The number of gaps filled is logged.
if ("paper_Tumor.stage" %in% names(clin)) {
  paper_stage <- blank_to_na(clin$paper_Tumor.stage)
  n_rescued <- sum(is.na(gdc$gdc_stage) & !is.na(paper_stage))
  gdc[, gdc_stage := fifelse(is.na(gdc_stage), paper_stage, gdc_stage)]
  log_msg("  stage: ", n_rescued, " gaps filled from paper_Tumor.stage")
}

# ---------------------------------------------------------------
# TCGA-CDR -- the endpoints, and the first source for age / sex / stage.
# ---------------------------------------------------------------
f_cdr <- file.path(RAW, "TCGA-CDR-SupplementalTableS1.xlsx")
if (!file.exists(f_cdr)) stop("missing ", f_cdr, "\n  run `make download` first")

log_msg("=== TCGA-CDR ===")
cdr <- as.data.table(suppressWarnings(readxl::read_excel(f_cdr, sheet = 1)))
# Column punctuation varies between distributions of this table
# (PFI.time / PFI_time / pfi time). All names are changed to lower case with
# underscores before anything is looked up, and the script stops if one of
# the columns it needs is not there.
setnames(cdr, tolower(gsub("[^A-Za-z0-9]+", "_", names(cdr))))
need <- c("bcr_patient_barcode", "type", "pfi", "pfi_time", "os", "os_time",
          "age_at_initial_pathologic_diagnosis", "gender",
          "ajcc_pathologic_tumor_stage")
absent <- setdiff(need, names(cdr))
if (length(absent))
  stop("TCGA-CDR: expected columns not found: ", paste(absent, collapse = ", "),
       "\n  present: ", paste(head(names(cdr), 40), collapse = ", "))

# The table covers all of TCGA; keep lung adenocarcinoma.
cdr <- cdr[type == "LUAD"]
log_msg("  LUAD patients: ", nrow(cdr))
cdr[, patient_barcode := patient_of(as.character(bcr_patient_barcode))]

# Remove patients whose data were withdrawn ("Redacted"), then check that
# every patient appears once.
if ("redaction" %in% names(cdr)) {
  n0 <- nrow(cdr)
  cdr <- cdr[is.na(redaction) | redaction != "Redacted"]
  log_msg("  after dropping redacted: ", nrow(cdr), " (dropped ", n0 - nrow(cdr), ")")
}
if (anyDuplicated(cdr$patient_barcode))
  stop("TCGA-CDR: duplicate patient barcodes after redaction filter")

# Age, sex and stage as TCGA-CDR records them.
cdr_cov <- cdr[, .(
  patient_barcode,
  cdr_age   = suppressWarnings(as.numeric(age_at_initial_pathologic_diagnosis)),
  cdr_sex   = tolower(blank_to_na(gender)),
  cdr_stage = blank_to_na(ajcc_pathologic_tumor_stage)
)]

# ---------------------------------------------------------------
# Covariates: TCGA-CDR first, colData second, the source recorded per field.
#
# The two tables are joined so that a patient in either one is kept
# (all = TRUE). coalesce_src() takes the first source's value if there is
# one and the second's otherwise, and notes which source was used. It logs
# how many values came from each source and how many are missing in both.
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

# The covariate table. For each of age, sex and stage: the value and its
# source. Stage and smoking are kept both as the original text (*_raw) and
# as numbers through the encodings above. sex_male is 1 for male, 0 for
# female.
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

# A stage or smoking text that is present but not in the encoding tables is
# a level this script has never seen. It becomes NA, and it is named in the
# log, so that the strongest covariate is not lost without notice.
unmapped <- setdiff(unique(na.omit(clinical$stage_raw)), names(STAGE_ORDINAL))
if (length(unmapped))
  log_msg("  !! unmapped stage levels -> NA: ", paste(unmapped, collapse = " | "))
unmapped_smk <- setdiff(unique(na.omit(clinical$smoking_raw)), names(SMOKING_EVER))
if (length(unmapped_smk))
  log_msg("  !! unmapped smoking levels -> NA: ", paste(unmapped_smk, collapse = " | "))

# ---------------------------------------------------------------
# Endpoints.
#
# PFI is the progression-free interval and OS is overall survival. For
# each there is an event column (1 = the event was seen, 0 = it was not)
# and a time column in days. A patient with event 0 is "censored": the
# follow-up ended before the event, so all that is known is that the
# patient was event-free up to that time.
#
# Times stay in days here, exactly as recorded. The analysis also stops
# every follow-up at 60 months (endpoint: admin_censor_months in the
# config; called administrative censoring because the cut is the same
# fixed time for everyone). That cut is applied when Python loads the data
# (src/luad/data.py), so that it is visible next to the models that use it.
#
# Patients without a usable PFI (no status, no time, or a time of zero)
# are dropped from this table.
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
# Write the two tables and report their size.
out_clin <- file.path(INTERIM, "clinical_raw.parquet")
out_endp <- file.path(INTERIM, "endpoints.parquet")
arrow::write_parquet(clinical,  out_clin)
arrow::write_parquet(endpoints, out_endp)

log_msg("=== written ===")
log_msg("  ", out_clin, "  (", nrow(clinical), " patients, ", ncol(clinical), " cols)")
log_msg("  ", out_endp, "  (", nrow(endpoints), " patients)")
log_msg("Next: Rscript prep/03_preprocess_rna.R")
