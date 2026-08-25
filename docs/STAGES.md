# Stage promotion: Stage 1 → Stage 2 → Stage 3

This project runs identically in all three stages. **Every promotion
below is a config/`.env` change only.** If a promotion ever seems to
require editing a file under `src/pdm/`, that's an architecture bug —
fix the abstraction, don't special-case the stage.

Current stage is tracked in [`PROJECT_NOTES.md`](../PROJECT_NOTES.md).

---

## Stage 1 — Build (dev environment)

- **Datasource**: local CSV (`data/raw/train_FD001.txt`, etc.)
- **Alerting**: console (loguru output)
- **Registry**: local MLflow (sqlite + local files), no server process
- **Install**: pip or conda, internet available

Setup:
```bash
make setup   # conda env create/update from environment.yml
make data    # ONE-TIME online step: downloads the C-MAPSS FD001 dataset
make train   # runs the full pipeline
make serve   # FastAPI on :8000
```

No `.env` file is needed — `csv` and `console` are the defaults and
require no credentials.

---

## Stage 2 — Office PWS (air-gapped test)

The office network has no internet and no `pip` — **conda-only**
installs from a pre-resolved channel/cache. Real machine data now comes
from the office's SQL Server/Postgres and/or MongoDB instead of sample
CSVs. Alerts now go to MS Teams as adaptive cards.

### What changes

1. **`configs/datasource_config.yaml`**
   ```yaml
   active_source: sql        # or: mongodb
   ```
   Fill in `sql:` (or `mongodb:`) connection details. Never hardcode
   credentials — set them via `.env` (copy `.env.example`):
   ```bash
   SQL_HOST=plant-sql-01.internal
   SQL_DB=machine_telemetry
   SQL_USER=svc_pdm
   SQL_PASSWORD=...
   ```

2. **`configs/alerting_config.yaml`**
   ```yaml
   active_channel: teams_webhook
   ```
   ```bash
   TEAMS_WEBHOOK_URL=https://<tenant>.webhook.office.com/webhookb2/...
   ```

3. **Environment install** — conda-only, no pip at any point after
   initial setup:
   ```bash
   conda env create -f environment.yml   # or conda env update
   ```
   (`environment.yml`'s `pip:` section installs `pandera` and
   `evidently`, which aren't reliably on conda-forge — this runs ONCE
   during setup while internet is still available. No pip call happens
   again after that.)

### What does NOT change

- Every file under `src/pdm/` — the `DataSource`, `Alerter`, and `Model`
  interfaces mean `pdm.features`, `pdm.models`, and `pdm.pipelines`
  never know or care which concrete backend is active.
- `configs/model_config.yaml` (unless you're deliberately retraining
  with different hyperparameters, which is a modeling decision, not a
  stage-promotion requirement).

### Validating the promotion

```bash
make test                       # unit tests still pass, no config needed
python -m pdm.pipelines.training_flow   # now ingests from SQL/Mongo
python scripts/validate_and_promote_model.py
```
`tests/integration/test_config_swap_datasource.py` proves programmatically
that flipping `active_source` changes which class `get_datasource()`
returns, with zero other code involved.

---

## Stage 3 — IIOT server (fully offline production)

No internet, **no installs of any kind** — not conda, not pip, nothing.
The exact environment from Stage 1/2 is copied over as a `conda-pack`
tarball and unpacked as-is.

### What changes

Same as Stage 2's config changes (datasource, alerting) if not already
applied there — Stage 3 typically inherits Stage 2's `configs/*.yaml`
verbatim, since the office PWS was the test bed for the same real data
sources.

Additionally, decide whether an MLflow *server* process runs on this
machine:
- **If yes** (still no server needed — `MlflowModelRegistry` talks
  directly to the local sqlite file, no `mlflow server` process
  required): no further change.
- **If you want zero MLflow dependency at inference time at all**:
  ensure a model has been promoted to
  `configs/model_config.yaml`'s `registry.local_fallback_path`
  (`pdm.registry.local_artifact.save_local_fallback`). `pdm.serving`
  already tries the MLflow registry first and falls back to this path
  automatically — see `src/pdm/serving/model_loader.py`.

### Deployment procedure

1. **On a machine with internet** (Stage 1 dev box or Stage 2 office
   PWS, once conda packages are cached): run
   ```bash
   bash scripts/build_bundle.sh
   ```
   This produces `bundle/pdm-bundle.tar.gz` — the packed conda
   environment plus `src/`, `configs/`, and `scripts/`.

2. **Copy the tarball to the IIOT server** via the site's approved
   offline transfer method (USB, internal file share — never a live
   network pull).

3. **On the IIOT server**, follow `UNPACK_AND_RUN.md` (included inside
   the bundle) — extract, `source env/bin/activate && conda-unpack`,
   set `PDM_CONFIG_DIR`/`PYTHONPATH`, and run the serving API or
   pipeline directly. No `conda`/`pip` command is ever invoked on this
   machine.

4. **Verify**: `curl http://localhost:8000/health` should report
   `model_loaded: true`, `datasource_reachable: true`.

### Anything that would silently need internet — call it out

- `scripts/download_data.py` — never runs on Stage 3; it's a Stage 1
  setup-only script.
- MLflow, Prefect, Evidently — all operate against local files only;
  none of them phone home or need a package registry at runtime.
- The MS Teams alerter needs outbound HTTPS to
  `*.webhook.office.com` specifically — if the IIOT server truly has
  zero outbound access (not even to the office's Teams tenant), switch
  `alerting_config.yaml` back to `console` and have a separate
  monitored process on a machine that *does* have that access forward
  the alert log, or accept alerts as a downstream (Qlik/Teams-poller)
  concern instead of an inline webhook call from the server itself.

---

## Rollback

Registry stage transitions are non-destructive — demoting the current
Production model is a single `MlflowModelRegistry.transition_stage(version,
"Archived")` call, and the previous Production version can be
re-promoted the same way. See `docs/runbook.md`.
