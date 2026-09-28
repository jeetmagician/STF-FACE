.DEFAULT_GOAL := help
SHELL := /bin/bash

BACKEND := backend
FRONTEND := frontend

.PHONY: help
help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

.PHONY: setup
setup: ## Install backend deps and download model weights
	cd $(BACKEND) && pip install -r requirements.txt
	cd $(BACKEND) && python scripts/download_models.py
	cd $(FRONTEND) && npm install

.PHONY: setup-insightface
setup-insightface: ## Add the InsightFace backend (NON-COMMERCIAL weights)
	@echo "NOTE: InsightFace pretrained weights are non-commercial research use only."
	cd $(BACKEND) && pip install -r requirements-insightface.txt
	cd $(BACKEND) && python scripts/download_models.py --backend insightface

.PHONY: dev-backend
dev-backend: ## Run the API with reload
	cd $(BACKEND) && uvicorn app.main:app --reload --port 8000

.PHONY: dev-frontend
dev-frontend: ## Run the Next.js dev server
	cd $(FRONTEND) && npm run dev

.PHONY: selfcheck
selfcheck: ## Fast pipeline smoke test (no weights, no pytest needed)
	cd $(BACKEND) && python scripts/selfcheck.py

.PHONY: test
test: ## Full backend test suite
	cd $(BACKEND) && pytest -v

.PHONY: typecheck
typecheck: ## Typecheck the frontend
	cd $(FRONTEND) && npm run typecheck

.PHONY: verify
verify: selfcheck test typecheck ## Everything

.PHONY: calibrate
calibrate: ## Fit calibration. Usage: make calibrate PAIRS=pairs.csv ROOT=/data
	cd $(BACKEND) && python scripts/fit_calibration.py $(PAIRS) --root $(ROOT) \
		--profile production --description "$(DESC)"

.PHONY: evaluate
evaluate: ## Evaluate. Usage: make evaluate PAIRS=pairs.csv ROOT=/data
	cd $(BACKEND) && python scripts/evaluate.py $(PAIRS) --root $(ROOT) \
		--plot roc.png --by-condition

.PHONY: up
up: ## Start everything with Docker
	docker compose up --build

.PHONY: down
down: ## Stop and remove containers
	docker compose down

.PHONY: clean
clean: ## Remove caches and build artefacts
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf $(BACKEND)/.pytest_cache $(FRONTEND)/.next
