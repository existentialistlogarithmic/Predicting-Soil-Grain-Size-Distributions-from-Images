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
	@test -f "$$HOME/.kaggle/kaggle.json" || \
	  (echo "No ~/.kaggle/kaggle.json - see README, 'Getting the data'"; exit 1)
	$(PYTHON) -m pip install --quiet kaggle
	mkdir -p $(DATA)
	kaggle competitions download -c $(COMP) -p $(DATA)
	cd $(DATA) && unzip -o -q $(COMP).zip && rm -f $(COMP).zip
	@echo "data ready in $(DATA)"

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
