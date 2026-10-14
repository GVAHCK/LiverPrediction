# Makefile for Referral-Budgeted, Sex-Audited Selective Classification
# "Referral-Budgeted, Sex-Audited Selective Classification for Small-Sample Liver Laboratory Data"

PYTHON ?= python

.PHONY: all env data audit nested policy stats robust hcv figs paper app test verify clean help

help:
	@echo "Available make targets:"
	@echo "  make env     - Install pinned dependencies"
	@echo "  make data    - Process and clean datasets (ILPD & UCI HCV)"
	@echo "  make audit   - Run dataset integrity and duplicate audit"
	@echo "  make nested  - Execute leak-free 5x3 nested cross-validation"
	@echo "  make policy  - Optimize referral-budgeted selective policy & baselines"
	@echo "  make stats   - Run patient-level paired bootstrap tests (2,000x)"
	@echo "  make robust  - Run duplicate sensitivity, missingness stress, and HCV replication"
	@echo "  make hcv     - Execute UCI HCV methodological replication"
	@echo "  make figs    - Generate publication vector figures and SHAP analysis"
	@echo "  make paper   - Auto-generate LaTeX tables and verify paper numbers"
	@echo "  make app     - Export v3.0.0 demonstration model bundle"
	@echo "  make test    - Run complete unit and integration test suite"
	@echo "  make all     - Execute end-to-end research and verification pipeline"

env:
	$(PYTHON) -m pip install -r requirements-lock.txt

data:
	$(PYTHON) -m src.data_audit

audit: data

nested:
	$(PYTHON) -m src.nested_selection

policy:
	$(PYTHON) -m src.selective_policy

stats:
	$(PYTHON) -m src.patient_statistics

robust:
	$(PYTHON) -m src.robustness_experiments

hcv:
	$(PYTHON) -m src.robustness_experiments

figs:
	$(PYTHON) -m src.interpretability_analysis

paper:
	$(PYTHON) -m src.generate_paper_tables
	$(PYTHON) -m src.verify_paper_numbers

app:
	$(PYTHON) -m src.export_production_model

test:
	$(PYTHON) -m pytest -v tests/

verify:
	$(PYTHON) -m src.verify_paper_numbers

all: data audit nested policy stats robust figs paper app test verify
	@echo "======================================================="
	@echo " [SUCCESS] Full research and verification pipeline complete!"
	@echo " Every table, figure, metric, and test has passed."
	@echo "======================================================="

clean:
	rm -rf __pycache__ src/__pycache__ tests/__pycache__ .pytest_cache
