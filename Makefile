# LUAD multi-omics integration. `make help` lists everything.
PY      := .venv/bin/python
SHELL   := /bin/bash
# A step fails if its script fails, not only if `tee` does.
.SHELLFLAGS := -o pipefail -c
# logs/ is not in the repository; the steps below write their logs there.
$(shell mkdir -p logs)
.PHONY: help setup test quick all cv bootstrap os external biology checks permute report figures cohort cytoscape prep download download-methylation freeze data

help:
	@echo "Analysis (in order; 'make all' runs them all: 2 h 52 min with 12 workers, measured 23 Nov 2026)"
	@echo "  make cv          1. nested CV, every arm and scenario   (37 min)"
	@echo "  make bootstrap   2. intervals for the fixed contrasts   (13 min)"
	@echo "  make os             steps 1-2 on the sensitivity endpoint (48 min)"
	@echo "  make external    3. TCGA -> GEO validation              (15 min)"
	@echo "  make biology     4. factors, gene lists, STRING network (under 2 min, needs internet)"
	@echo "  make cytoscape      send the network to Cytoscape: style, analyse, export (under 2 min)"
	@echo "  make permute        step 1 with a shuffled outcome      (10 min)"
	@echo "  make checks      5. the four robustness checks          (46 min)"
	@echo "  make report      6. write RESULTS.md from the tables    (seconds)"
	@echo "  make figures     7. publication figures and source data (20 seconds)"
	@echo "  make cohort      8. baseline table and flow of the patients analysed (seconds)"
	@echo "A failed step stops the run; continue with the remaining targets, for example"
	@echo "  make biology cytoscape permute checks report figures cohort"
	@echo ""
	@echo "Quick"
	@echo "  make test        unit tests on synthetic data (10 seconds, no data needed)"
	@echo "  make quick       smoke test of step 1 on real data (1 minute)"
	@echo ""
	@echo "Setup and data"
	@echo "  make setup       create .venv and install requirements"
	@echo "  make data        download the prepared data (1.5 GB) and check it; then 'make all' runs"
	@echo "  make prep        rebuild data/processed from data/raw (needs R, see README)"
	@echo "  make download    fetch the raw data (needs R; skips what is present)"

setup:
	python3 -m venv .venv
	$(PY) -m pip install -q --upgrade pip
	$(PY) -m pip install -q -r requirements.txt

test:
	$(PY) -m pytest tests -q

quick:
	$(PY) scripts/01_cv.py --quick

cv:
	$(PY) scripts/01_cv.py 2>&1 | tee logs/01_cv_pfi.log
bootstrap:
	$(PY) scripts/02_bootstrap.py 2>&1 | tee logs/02_bootstrap_pfi.log
os:
	$(PY) scripts/01_cv.py --endpoint os 2>&1 | tee logs/01_cv_os.log
	$(PY) scripts/02_bootstrap.py --endpoint os 2>&1 | tee logs/02_bootstrap_os.log
external:
	$(PY) scripts/03_external.py 2>&1 | tee logs/03_external.log
biology:
	$(PY) scripts/04_biology.py 2>&1 | tee logs/04_biology.log
cytoscape:
	$(PY) scripts/04b_cytoscape.py 2>&1 | tee logs/04b_cytoscape.log
permute:
	$(PY) scripts/01_cv.py --permute --arms clinical early mofa 2>&1 | tee logs/01_cv_permuted.log
checks:
	$(PY) scripts/05_checks.py 2>&1 | tee logs/05_checks.log
report:
	$(PY) scripts/06_report.py
figures:
	$(PY) scripts/07_figures.py
cohort:
	$(PY) scripts/08_cohort_table.py

all: test cv bootstrap os external biology cytoscape permute checks report figures cohort

freeze:
	$(PY) -m pip freeze > results/pip_freeze.txt

# ---- the prepared data, as one archive. DATA_URL is the file's address in the
# data record (see README, "Data"); the checksum is the one in config/.
DATA_URL ?=
DATA_TAR := luad-multiomics-data-v1.tar
data:
	@test -n "$(DATA_URL)" || { echo "Set DATA_URL to the address of $(DATA_TAR) (README, section Data)."; exit 1; }
	curl -L --fail -o $(DATA_TAR) "$(DATA_URL)"
	sha256sum -c config/data_bundle.sha256
	tar -xf $(DATA_TAR)
	rm $(DATA_TAR)
	@echo "data/processed and data/interim/endpoints.parquet are in place"

# ---- data preparation (R + Python). Only needed to rebuild data/ from scratch.
download:
	Rscript prep/01_download.R
download-methylation:
	Rscript -e 'suppressPackageStartupMessages(library(TCGAbiolinks)); \
	  dir.create("data/raw/methylation", recursive=TRUE, showWarnings=FALSE); \
	  setwd("data/raw/methylation"); \
	  q <- GDCquery(project="TCGA-LUAD", data.category="DNA Methylation", \
	    platform="Illumina Human Methylation 450", \
	    data.type="Methylation Beta Value", sample.type="Primary Tumor"); \
	  GDCdownload(q, method="api", files.per.chunk=20, directory="GDCdata")'
prep:
	Rscript prep/01b_filter_geo.R
	Rscript prep/utils/export_barcodes.R
	Rscript prep/utils/build_pathways.R
	Rscript prep/03_preprocess_rna.R
	$(PY) prep/04_preprocess_mutation.py
	Rscript prep/utils/export_clinical.R
	$(PY) prep/05_preprocess_clinical.py
	Rscript prep/utils/export_geo.R
	Rscript prep/16_preprocess_methylation.R
