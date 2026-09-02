# FX Triangular Pricing
#
#
#   make            print this list
#   make all        gate, analysis, report
#   make data       the 1-second grid only
#   make check      the data-quality gate; exit code propagates, for CI
#   make analysis   01 to 07, skipping the gate
#   make report     compile the PDF
#   make test       pytest
#   make fast       analysis with 02's null calibration switched off
#   make clean      remove generated output, keep the parsed grid
#   make distclean  remove the parsed grid too

PY      ?= python
PYTEST  ?= pytest
LATEXMK ?= latexmk

SCRIPTS ?= scripts
TESTS   ?= tests
RAW     ?= data/raw
PROC    ?= data/processed
OUT     ?= output
REPORT  ?= report
#
#   make all ACCEPT_KNOWN_BIAS=1
#
ACCEPT_KNOWN_BIAS ?=

# --------------------------------------------------------------- artefacts

GRID    := $(PROC)/synchronized_rates.parquet
CLEAN   := $(OUT)/data/01_clean.parquet
REGIMES := $(OUT)/data/02_regimes.parquet
VAR     := $(OUT)/data/03_var_model.parquet
GRANGER := $(OUT)/data/04_granger_tests.parquet
BOUNDS  := $(OUT)/data/05_bounds.parquet
RECON   := $(OUT)/tables/06_closure_reconciliation.csv
SWEEP   := $(OUT)/tables/07_boundary_named.csv
PDF     := $(REPORT)/main.pdf

STAMP   := $(OUT)/.quality-ok

UTILS   := $(SCRIPTS)/utils.py
RAWDATA := $(wildcard $(RAW)/histdata/*.csv) $(wildcard $(RAW)/dukascopy/*.csv)

ANALYSIS := $(CLEAN) $(REGIMES) $(VAR) $(GRANGER) $(BOUNDS) $(RECON) $(SWEEP)

# A half-written parquet is newer than its inputs and would be treated as up
# to date on the next run. Delete the target if its recipe fails.
.DELETE_ON_ERROR:

.PHONY: help all data check analysis report test fast clean distclean

help:
	@sed -n '3,17p' $(MAKEFILE_LIST) | sed 's/^# \{0,1\}//'

all: $(STAMP) analysis report

data: $(GRID)

analysis: $(ANALYSIS)

report: $(PDF)

# ------------------------------------------------------------------- stages

$(GRID): $(SCRIPTS)/00_ingest_and_sync.py $(RAWDATA)
	$(PY) $(SCRIPTS)/00_ingest_and_sync.py

# Phony rather than file-backed: the exit code is the product, and CI wants it
# on every invocation rather than only when the inputs move.
check: $(GRID)
	$(PY) $(SCRIPTS)/00b_data_quality.py

# The same gate, recorded, so `all` runs it once and the analysis can depend
# on it having been considered.
$(STAMP): $(GRID) $(SCRIPTS)/00b_data_quality.py $(UTILS)
	@mkdir -p $(dir $@)
	@$(PY) $(SCRIPTS)/00b_data_quality.py; status=$$?; \
	if [ $$status -ne 0 ] && [ -z "$(ACCEPT_KNOWN_BIAS)" ]; then \
	  echo ""; \
	  echo "  The data-quality gate failed (exit $$status)."; \
	  echo "  This sample fails it by design; see README section 2."; \
	  echo "  To proceed anyway:  make all ACCEPT_KNOWN_BIAS=1"; \
	  exit $$status; \
	fi
	@touch $@

$(CLEAN): $(SCRIPTS)/01_exploratory_analysis.py $(UTILS) $(GRID)
	$(PY) $(SCRIPTS)/01_exploratory_analysis.py

$(REGIMES): $(SCRIPTS)/02_changepoint_detection.py $(UTILS) $(CLEAN)
	$(PY) $(SCRIPTS)/02_changepoint_detection.py

$(VAR): $(SCRIPTS)/03_var_analysis.py $(UTILS) $(CLEAN) $(REGIMES)
	$(PY) $(SCRIPTS)/03_var_analysis.py

# 04 loads 03 by path for its row-admission rules, so it depends on that
# source file and not only on 03's output.
$(GRANGER): $(SCRIPTS)/04_granger_causality.py $(SCRIPTS)/03_var_analysis.py \
            $(UTILS) $(CLEAN) $(REGIMES)
	$(PY) $(SCRIPTS)/04_granger_causality.py

# 05 reads the quoted grid rather than 01's mids: it needs bid and ask.
$(BOUNDS): $(SCRIPTS)/05_execution_bounds.py $(UTILS) $(GRID) $(REGIMES)
	$(PY) $(SCRIPTS)/05_execution_bounds.py

# 06 reads tables rather than data and resolves its input directory against
# the working directory, so both paths are passed explicitly.
$(RECON): $(SCRIPTS)/06_reconciliation.py $(UTILS) $(CLEAN) $(REGIMES) $(VAR)
	$(PY) $(SCRIPTS)/06_reconciliation.py $(OUT)/tables $(OUT)/tables

$(SWEEP): $(SCRIPTS)/07_boundary_sweep.py $(UTILS) $(CLEAN) $(REGIMES)
	$(PY) $(SCRIPTS)/07_boundary_sweep.py

# ------------------------------------------------------------------- report

$(PDF): $(REPORT)/main.tex $(ANALYSIS) \
        $(wildcard $(REPORT)/tables/*.tex) $(wildcard $(OUT)/figures/*.png)
	$(LATEXMK) -pdf -silent -cd $(REPORT)/main.tex

# --------------------------------------------------------------------- misc

test:
	$(PYTEST) $(TESTS) -q

fast:
	CP_NULL_DRAWS=0 $(MAKE) analysis

clean:
	rm -rf $(OUT)/tables/* $(OUT)/figures/* $(OUT)/data/* $(REPORT)/tables/*
	rm -f $(STAMP)
	$(LATEXMK) -C -cd $(REPORT)/main.tex 2>/dev/null || true

distclean: clean
	rm -f $(GRID)