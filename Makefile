PYTHON ?= python3
COMP   := soil-grain-size-from-photos
DATA   ?= data/raw

.PHONY: help setup download synthetic features cv submit validate test clean

help:
	@echo "setup      install the pipeline and its dependencies"
	@echo "download   fetch the competition data (needs ~/.kaggle/kaggle.json)"
	@echo "synthetic  generate a fake dataset so the pipeline runs without Kaggle"
	@echo "features   extract and cache photo features for both splits"
	@echo "cv         cross-validate every model against the competition metric"
	@echo "submit     fit on all data and write artifacts/submissions/<model>.csv"
	@echo "validate   check a submission: make validate SUBMISSION=path/to.csv"
	@echo "test       run the test suite"

setup:
	$(PYTHON) -m pip install -e ".[dev]"

download:
	./scripts/download_data.sh $(DATA)

synthetic:
	$(PYTHON) -m soilgsd make-synthetic --out data/synthetic

features:
	$(PYTHON) -m soilgsd features

cv:
	$(PYTHON) -m soilgsd cv

submit:
	$(PYTHON) -m soilgsd submit

validate:
	@test -n "$(SUBMISSION)" || (echo "usage: make validate SUBMISSION=path/to.csv"; exit 1)
	$(PYTHON) -m soilgsd validate "$(SUBMISSION)" --template $(DATA)/sample_submission.csv

test:
	$(PYTHON) -m pytest

clean:
	rm -rf artifacts/features artifacts/cv .pytest_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
