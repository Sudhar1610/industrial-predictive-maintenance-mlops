# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

An end-to-end predictive maintenance MLOps platform for industrial turbofan
assets (NASA C-MAPSS FD001 dataset as the industrial-credible stand-in for
real plant telemetry). Two prediction heads share one pipeline: binary
failure classification and remaining-useful-life (RUL) regression. Trained
models are tracked in a local MLflow registry, served via FastAPI, and
monitored for drift.

Read `README.md` first for the architecture diagram and results table — it
is kept accurate and is not duplicated here.

## The defining constraint: three deployment stages, one codebase

This codebase must run unmodified in three environments — a laptop
(Stage 1), an air-gapped office network (Stage 2), and a fully offline
industrial server (Stage 3). **Moving between stages is a config/`.env`
change, never a code change.** The current stage is tracked in
`PROJECT_NOTES.md`; the full promotion procedure is in `docs/STAGES.md`.

**If a change to move stages ever requires editing a file under `src/pdm/`,
that is an architecture bug** — fix the abstraction (interface + factory),
don't special-case the stage in application code.

This is enforced by a strict interface/factory pattern used identically in
three places:

- `pdm.data`: `DataSource` interface (`src/pdm/data/base.py`) with
  `CsvDataSource`, `SqlDataSource`, `MongoDataSource` implementations,
  selected by `pdm.data.factory.get_datasource()` reading
  `configs/datasource_config.yaml`'s `active_source`.
- `pdm.models`: `Model` interface (`src/pdm/models/base.py`, dual-head
  `fit`/`predict`/`save`/`load`) with sklearn RandomForest, XGBoost, and
  PyTorch LSTM implementations, selected via `configs/model_config.yaml`'s
  `active_model`.
- `pdm.alerting`: `Alerter` interface with console (loguru) and MS Teams
  webhook implementations, selected via `configs/alerting_config.yaml`'s
  `active_channel`.

`pdm.features`, `pdm.models`, and `pdm.pipelines` never import a concrete
`CsvDataSource`/`SqlDataSource`/`ConsoleAlerter`/`TeamsAlerter` — only the
interface types, obtained from the relevant factory. When adding a new
backend for any of these three, add a class implementing the interface,
wire it into the factory's dispatch, and add the config shape to
`pdm/config/schemas.py` — do not add branching elsewhere.

Model serialization (`Model.save`/`load`) must not depend on any running
service (no MLflow server, no network) — this is what makes Stage 3
offline inference possible. `pdm.serving.model_loader.load_production_model`
tries the MLflow registry first and transparently falls back to
`pdm.registry.local_artifact` if the registry is unavailable or has no
Production model; both paths return the same `Model` type, so which one
actually loaded is invisible to `pdm.serving.app`.

Config files (`configs/*.yaml`) are pydantic-validated
(`pdm/config/schemas.py`, loaded via `pdm/config/settings.py`).
Credentials are never hardcoded in YAML — they're interpolated from
environment variables (`${SQL_PASSWORD}` etc.) at load time; copy
`.env.example` to `.env` for Stage 2/3 secrets.

## Commands

```bash
make setup     # conda env create/update from environment.yml
make data      # one-time online step: download C-MAPSS FD001 into data/raw/
make train     # full Prefect pipeline: ingest -> validate -> feature -> train -> evaluate -> register
make serve     # FastAPI on :8000 (uvicorn --reload)
make up        # full local stack via docker-compose: mlflow, prefect, prometheus, grafana, api
make down      # tear down the docker-compose stack
make test      # pytest (coverage report to terminal)
make lint      # ruff check + ruff format --check
make format    # ruff format + ruff check --fix
make typecheck # mypy src
make bundle    # produce the Stage 3 conda-pack tarball via scripts/build_bundle.sh

python scripts/validate_and_promote_model.py   # CI gate: promote newly-trained model to Production only if it doesn't regress vs current
```

Single test / single file:
```bash
pytest tests/unit/test_features.py
pytest tests/unit/test_features.py::test_rolling_mean_window_size
pytest -k "drift"
```

CI (`.github/workflows/ci.yml`) runs lint → typecheck/test (in parallel,
both depend on lint) with `pip install -e ".[dev,sql]"`, and enforces
`--cov-fail-under=80`. `.github/workflows/model_validation.yml` runs the
full pipeline end to end: download data, train, validate/promote, build
the serving Docker image, start it, and hit `/health` and `/predict` for
real — treat this workflow as the source of truth for expected
`/predict` response shape when changing `pdm.serving`.

## Code conventions

- Python 3.11 only (`requires-python = ">=3.11,<3.12"`).
- `ruff` line length 100; `E501` (line length) and `N806`/`N803` are
  deliberately ignored — `X`/`X_train`-style names for feature matrices
  are the standard sklearn/numpy convention used throughout, not a lint
  violation to fix.
- `mypy` runs with `disallow_untyped_defs = true` — new functions need
  type annotations.
- Two pins in `pyproject.toml` carry explanatory comments; do not casually
  bump them: `anyio==4.4.0` (prefect 2.20.2's task-group wrapper breaks on
  anyio>=4.5) and `pathspec==0.12.1` (dvc 3.53.2 imports a symbol removed
  in pathspec's newer 1.x series).
- `tests/conftest.py` provides `synthetic_cmapss_df`/`cmapss_csv_dir`
  fixtures — a small deterministic synthetic dataset shaped like real
  C-MAPSS data, for fast unit tests that don't need the ~real downloaded
  dataset. Prefer these over hitting `data/raw/` in unit tests;
  `tests/integration/` is where tests exercise the real pipeline
  end-to-end.
- `scripts/download_data.py` is the *only* step in the entire system
  that requires internet outside of environment setup — never add a
  second one. Anything that would silently need network access at
  runtime (a new external API call, a package fetch) breaks the Stage 3
  offline guarantee and needs a config-gated alternative instead.

## Project layout

```
configs/        pydantic-validated YAML: datasource, alerting, model, logging
src/pdm/
  data/          DataSource interface + CSV/SQL/MongoDB implementations
  features/      rolling stats, lags, degradation slopes, labels, sequences
  models/        Model interface + sklearn/XGBoost/LSTM backends
  registry/      local MLflow registry + no-server local artifact fallback
  evaluation/    metrics (incl. NASA asymmetric RUL score), threshold tuning, drift
  serving/       FastAPI inference API
  alerting/      Alerter interface + console/Teams implementations
  monitoring/    prediction logging, drift job, Prometheus metrics
  pipelines/     Prefect training flow (the orchestrator tying every module above together)
tests/           unit/ (fast, synthetic fixtures) + integration/ (pytest)
docker/          multi-stage Dockerfiles + docker-compose (local stack)
scripts/         download_data.py (the only online step), build_bundle.sh, validate_and_promote_model.py (CI gate)
docs/            architecture.md, STAGES.md (promotion guide), AWS_MIGRATION.md, adr/ (decision records)
```

`docs/STAGES.md` and `PROJECT_NOTES.md` are living documents — when a
stage promotion actually happens (real Stage 2/3 run), update
`PROJECT_NOTES.md`'s log and current-stage header.
