.PHONY: setup data train test serve up down lint bundle format typecheck

# Create the conda env from environment.yml (or fall back to pip -e .[dev] in CI).
setup:
	conda env create -f environment.yml || conda env update -f environment.yml

# One-time online step: download and cache C-MAPSS FD001 into data/raw/.
data:
	python scripts/download_data.py

# Run the full Prefect training pipeline: ingest -> validate -> feature -> train -> evaluate -> register.
train:
	python -m pdm.pipelines.training_flow

lint:
	ruff check src tests
	ruff format --check src tests

format:
	ruff format src tests
	ruff check --fix src tests

typecheck:
	mypy src

test:
	pytest

# Serve the FastAPI inference API locally.
serve:
	uvicorn pdm.serving.app:app --host 0.0.0.0 --port 8000 --reload

# Bring up the full local stack: mlflow, prefect server, prometheus, grafana, api.
up:
	docker compose -f docker/docker-compose.yml up -d --build

down:
	docker compose -f docker/docker-compose.yml down

# Produce the conda-pack tarball for Stage 3 handoff.
bundle:
	bash scripts/build_bundle.sh
