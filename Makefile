# LUAD multi-omics integration. `make help` lists everything.
#
# A Makefile gives short names ("targets") to commands. `make cv` runs the
# commands written under `cv:`. Run every target from the project folder.
#
# PY is the Python of the project's own environment (.venv, created by
# `make setup`). SHELL makes every command run in bash.
PY      := .venv/bin/python
SHELL   := /bin/bash
# A step fails if its script fails, not only if `tee` does.
.SHELLFLAGS := -o pipefail -c
# logs/ is not in the repository; the steps below write their logs there.
$(shell mkdir -p logs)
.PHONY: help setup test quick all cv bootstrap os external biology checks permute report figures cohort verify survival factor cytoscape prep download download-methylation freeze data

# ---- help: print the list of targets with what they do and how long they
# take. It runs nothing. Start here.
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
	@echo "  make verify      9. compare your tables with the published ones (a second)"
	@echo "  make survival   10. survival curves by predicted risk; added after the first run (seconds)"
	@echo "  make factor     11. what MOFA factor 2 is; added after the first run (seconds)"
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
	@echo "  make download    fetch the raw data except methylation (needs R; skips what is present)"
	@echo "  make download-methylation   fetch the methylation files and the probe annotation (needs R)"

# ---- setup: run once, first. Creates the Python environment in .venv and
# installs the packages of requirements.txt. Needs python3 and the internet.
setup:
	python3 -m venv .venv
	$(PY) -m pip install -q --upgrade pip
	$(PY) -m pip install -q -r requirements.txt

# ---- test and quick: two fast checks that the installation works.
# `make test` runs the unit tests on made-up data and needs no data files.
# `make quick` runs step 1 on the real data with tiny settings.
test:
	$(PY) -m pytest tests -q

quick:
	$(PY) scripts/01_cv.py --quick

# ---- the analysis, one target per step, in the order they run.
# Run one of these to repeat a single step, or to continue after a step
# that failed. Each needs the prepared data (`make data`, or `make download`
# and `make prep`) and the results of the steps before it. The longer steps
# also write what they print to a file in logs/.
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
# Steps 10 and 11 were added after the first run (see README). They read the
# predictions of step 1 and the factor scores of step 4, and fit nothing.
survival:
	$(PY) scripts/10_survival_curves.py
	$(PY) scripts/10_survival_curves.py --endpoint os
factor:
	$(PY) scripts/11_factor_biology.py
verify:
	$(PY) scripts/09_verify.py

# ---- all: the whole analysis, in order. A step that fails stops the run.
all: test cv bootstrap os external biology cytoscape permute checks survival factor report cohort figures verify

# ---- freeze: write the exact versions of the installed Python packages to
# requirements.lock. Run it after a full run to record the environment.
freeze:
	$(PY) -m pip freeze > requirements.lock

# ---- data: download the prepared data as one archive. This is the short
# way to get the data; R is not needed. Run it once, before `make all`:
#   make data DATA_URL=<address of the archive>
# DATA_URL is the file's address in the data record (see README, "Data").
# The archive is checked against the checksum in config/data_bundle.sha256
# before it is unpacked, and the downloaded archive is deleted afterwards.
DATA_URL ?=
DATA_TAR := luad-multiomics-data-v1.tar
data:
	@test -n "$(DATA_URL)" || { echo "Set DATA_URL to the address of $(DATA_TAR) (README, section Data)."; exit 1; }
	curl -L --fail -o $(DATA_TAR) "$(DATA_URL)"
	sha256sum -c config/data_bundle.sha256 || shasum -a 256 -c config/data_bundle.sha256
	tar -xf $(DATA_TAR)
	rm $(DATA_TAR)
	@echo "data/processed and data/interim/endpoints.parquet are in place"

# ---- download, download-methylation, prep: the long way to get the data.
# Only needed to rebuild data/ from the public sources (R and Python,
# about 10 GB; see README, "Data"). The R packages are installed with
# `Rscript prep/00_setup_r.R`. Run the three targets in this order:
#   make download              the raw data except methylation
#   make download-methylation  the methylation files of TCGA-LUAD
#   make prep                  the nine preparation steps, in order
# Each prep script says at its top what it reads and writes.
# prep/16_preprocess_methylation.R also needs a probe annotation file,
# data/raw/methylation/HM450.hg38.manifest.tsv.gz, which no target fetches.
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
	# The probe annotation (which probe sits where, and which probes to mask),
	# from the Zhou lab's InfiniumAnnotation collection. Checked against the
	# checksum of the copy the published data were made with.
	curl -L --fail -o data/raw/methylation/HM450.hg38.manifest.tsv.gz \
	  https://github.com/zhou-lab/InfiniumAnnotationV1/raw/main/Anno/HM450/HM450.hg38.manifest.tsv.gz
	echo "3b2e1598d456b12b38da9bc6d340be59  data/raw/methylation/HM450.hg38.manifest.tsv.gz" | md5sum -c -
# The nine steps of prep, in an order where every step finds the files of the
# steps before it. export_clinical.R comes before the mutation step because
# that step reads the file it writes (data/interim/clinical_raw.parquet).
prep:
	Rscript prep/01b_filter_geo.R
	Rscript prep/utils/export_barcodes.R
	Rscript prep/utils/build_pathways.R
	Rscript prep/03_preprocess_rna.R
	Rscript prep/utils/export_clinical.R
	$(PY) prep/04_preprocess_mutation.py
	$(PY) prep/05_preprocess_clinical.py
	Rscript prep/utils/export_geo.R
	Rscript prep/16_preprocess_methylation.R
