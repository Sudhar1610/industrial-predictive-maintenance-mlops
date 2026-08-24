# Project Notes

## Current Stage: **Stage 1 — Build (dev environment, local CSV)**

Started: 2026-08-24

## Stage definitions

- **Stage 1 (here now):** local dev, C-MAPSS FD001 CSV via `CsvDataSource`,
  console alerting, local MLflow (sqlite+file), no Docker services required
  to run tests. GitHub-ready: full docs, clean architecture.
- **Stage 2 (office PWS, air-gapped):** `datasource_config.yaml
  active_source: sql|mongodb`, `alerting_config.yaml active_channel:
  teams_webhook`. Conda-only install (no pip at runtime). Connects to real
  machine data.
- **Stage 3 (IIOT server, fully offline):** same code + `conda-pack`
  bundle from `scripts/build_bundle.sh` copied over. No internet, no
  installs. Model loaded via `local_artifact.py` with no MLflow server
  running. Writes results back to the database.

## Promotion checklist (Stage 1 -> Stage 2)

See `docs/STAGES.md` for the full procedure. Summary: only
`configs/*.yaml` + `.env` change; zero application-code edits.

## Log

- 2026-08-24: Repo scaffolded. Architecture, config shapes, and tech stack
  approved. Beginning implementation in order: data -> features -> models
  -> registry -> evaluation -> serving -> alerting -> monitoring ->
  pipeline -> CI/CD -> bundle -> docs.
- 2026-08-24: Stage 1 build complete. All modules implemented and tested
  end-to-end against real C-MAPSS FD001 data, including a live
  train -> register -> promote -> serve -> /predict -> /metrics run over
  HTTP. 127 tests, 96% coverage, ruff/mypy clean. DVC wired up with a
  local remote (push/pull round-trip verified). Full docs, ADRs, Docker,
  CI/CD (incl. the model-validation promotion gate), and the Stage 3
  conda-pack bundle script are all in place.
  Two real bugs found and fixed by actually running the pipeline rather
  than trusting it from tests alone: (1) prefect 2.20.2 breaks against
  anyio>=4.5, pinned anyio==4.4.0; (2) MlflowModelRegistry didn't create
  its sqlite db's parent directory on a fresh checkout, fixed with a
  regression test; (3) dvc 3.53.2 breaks against pathspec's newer 1.x
  series, pinned pathspec==0.12.1.
  Not yet done: no real Stage 2 (SQL/MongoDB) or Stage 3 (IIOT server)
  run has happened -- those implementations are built and unit-tested
  against mocks, but unverified against a real database or a real
  offline server. That's the natural next milestone whenever real
  machine data / office PWS access is available.
