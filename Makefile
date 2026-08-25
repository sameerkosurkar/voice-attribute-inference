# Convenience targets. Nothing here is required -- every command is a plain
# docker/pytest/python invocation, spelled out in the README.

PY      ?= python3
VENV    ?= .venv
BIN     := $(VENV)/bin
SAMPLES ?= samples
ONNX    ?= models/onnx/age_gender.onnx

.DEFAULT_GOAL := help
.PHONY: help venv install sample run test test-slow test-all verify export-onnx smoke ws eval docker-build docker-up docker-down docker-logs clean

help: ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
	  | awk 'BEGIN{FS=":.*?## "};{printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

venv: ## Create the virtualenv
	$(PY) -m venv $(VENV)

install: venv ## Install dependencies (CPU-only torch)
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install --index-url https://download.pytorch.org/whl/cpu \
	    torch==2.11.0 torchaudio==2.11.0
	$(BIN)/pip install -r requirements-dev.txt

sample: ## Generate audio fixtures (offline, no dataset needed)
	$(BIN)/python scripts/make_sample_audio.py --outdir $(SAMPLES)

run: ## Run the service locally on :8000
	$(BIN)/python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --no-access-log

test: ## Fast suite (mock backend, no weights)
	$(BIN)/python -m pytest -q -m "not slow"

test-slow: ## Real-weights suite, incl. latency + label-mapping checks
	VA_ONNX_PATH=$(ONNX) $(BIN)/python -m pytest -q -m slow -s

test-all: test test-slow ## Everything

verify: ## Check the gender label mapping against known-gender audio
	$(BIN)/python scripts/verify_gender_mapping.py

export-onnx: ## Export the ONNX graph locally (enables the parity tests)
	$(BIN)/python scripts/export_onnx.py --out $(ONNX)

smoke: ## End-to-end smoke test against a running service
	./scripts/smoke_test.sh

ws: ## Streaming demo against a running service
	$(BIN)/python scripts/ws_client.py

eval: ## Eval harness (override ARGS=...)
	$(BIN)/python eval/run_eval.py $(ARGS)

docker-build: ## Build the image
	docker compose build

docker-up: ## Start the service
	docker compose up -d

docker-down: ## Stop and remove
	docker compose down

docker-logs: ## Tail logs
	docker compose logs -f

clean: ## Remove generated artefacts
	rm -rf $(SAMPLES) models .pytest_cache **/__pycache__ .coverage
