# Moltbook Human-Like Outlier Detection Pipeline
# ================================================
# Run `make all` for the full pipeline, or individual targets as needed.
# Prerequisites: Python 3.10+, internet access for first run.

PYTHON ?= .venv/bin/python
CONFIG ?= config.yaml

.PHONY: setup data validate features outliers analyze audit sensitivity paper all clean help

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

setup: ## Create venv, install deps, download NLTK data
	python3 -m venv .venv
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -c "import nltk; nltk.download('punkt_tab', quiet=True)" 2>/dev/null || $(PYTHON) -c "import nltk; nltk.download('punkt', quiet=True)"
	$(PYTHON) -c "import nltk; nltk.download('words', quiet=True)"
	@echo "✓ Setup complete. Activate with: source .venv/bin/activate"

data: data/raw/posts.parquet ## Stage 1: Download Moltbook dataset
data/raw/posts.parquet:
	$(PYTHON) scripts/01_download_data.py --config $(CONFIG)

validate: data outputs/validate/data_profile.json ## Stage 2: Validate and profile the raw data
outputs/validate/data_profile.json: data/raw/posts.parquet
	$(PYTHON) scripts/02_validate_data.py --config $(CONFIG)

features: validate outputs/features/features.parquet ## Stage 3: Extract linguistic features
outputs/features/features.parquet: data/processed/posts_clean.parquet
	$(PYTHON) scripts/03_build_features.py --config $(CONFIG)

outliers: features outputs/outliers/outliers.parquet ## Stage 4: Run ensemble outlier detection
outputs/outliers/outliers.parquet: outputs/features/features.parquet
	$(PYTHON) scripts/04_detect_outliers.py --config $(CONFIG)

analyze: outliers ## Stage 5: Generate summary tables and plots
	$(PYTHON) scripts/05_analyze_results.py --config $(CONFIG)

audit: outliers ## Stage 6: Generate audit samples for human review
	$(PYTHON) scripts/06_make_audit_samples.py --config $(CONFIG)

sensitivity: features ## Stage 7: Sensitivity analysis across thresholds
	$(PYTHON) scripts/07_sensitivity.py --config $(CONFIG)

paper: analyze ## Stage 8: Generate arXiv LaTeX paper from results
	$(PYTHON) scripts/08_generate_paper.py --config $(CONFIG)

all: data validate features outliers analyze audit sensitivity paper ## Run full pipeline end-to-end

clean: ## Remove generated outputs (keeps raw data)
	rm -rf outputs/ data/processed/ logs/
	@echo "✓ Cleaned outputs, processed data, and logs"

distclean: clean ## Also remove raw data and venv
	rm -rf data/raw/ .venv/
	@echo "✓ Cleaned everything including raw data and venv"
