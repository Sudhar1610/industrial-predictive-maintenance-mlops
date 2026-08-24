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
