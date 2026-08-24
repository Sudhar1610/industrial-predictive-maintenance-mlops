# Runbook

Operational procedures for running this system day-to-day, once it's
past initial setup. For first-time setup, see the README's Quickstart;
for moving between deployment stages, see `docs/STAGES.md`.

## Training a new model

```bash
make train
# equivalent to: python -m pdm.pipelines.training_flow
```

This ingests from whichever `datasource_config.yaml` currently points
at, validates, engineers features, splits by unit, trains, evaluates,
and registers the result as a new **Staging** version in the local
MLflow registry. It does **not** touch Production.

## Promoting a model to Production

```bash
python scripts/validate_and_promote_model.py
```

Compares the latest Staging version's guardrail metric
(`classification.target_metric` in `configs/model_config.yaml`, default
`f1`) against the current Production version. Promotes if it doesn't
regress by more than the tolerance (`--tolerance`, default 2%); exits 1
and leaves Production untouched otherwise. This is what CI's
`model_validation.yml` workflow runs automatically on every push.

To promote manually regardless of the guardrail (e.g. after manual
review of a borderline case):
```python
from pdm.config import settings
from pdm.registry.mlflow_registry import MlflowModelRegistry

registry = MlflowModelRegistry(settings.get_model_config().registry)
registry.transition_stage("<version>", "Production")
```

## Rolling back a bad Production model

Every previous version is still in the registry (archived, not
deleted). To roll back:
```python
from pdm.config import settings
from pdm.registry.mlflow_registry import MlflowModelRegistry

registry = MlflowModelRegistry(settings.get_model_config().registry)
registry.transition_stage("<previous_good_version>", "Production")
```
This automatically archives whatever was in Production. Restart the
serving process (or wait for its next `/health`-driven restart, if one
is configured) to pick up the change — `pdm.serving.app`'s `lifespan`
hook only loads the Production model once at startup, not per-request.

## Checking system health

```bash
curl http://localhost:8000/health
```
```json
{"status": "ok", "model_loaded": true, "model_backend": "xgboost", "datasource_reachable": true}
```
`status: degraded` means either no model is loaded or the configured
datasource failed its health check — check the API's logs for which.

## Investigating a drift alert

1. Find the alert's `Alert.details` — includes `share_of_drifted_columns`
   and which sensor columns were checked.
2. Re-run the check manually for the full report (not just the
   summary that went into the alert):
   ```python
   from pdm.config import settings
   from pdm.data.factory import get_datasource
   from pdm.monitoring.drift_job import run_drift_check

   ds_config = settings.get_datasource_config()
   model_config = settings.get_model_config()
   reference_df = get_datasource(ds_config).fetch_training_data()

   result = run_drift_check(
       reference_df, model_config.feature_engineering, settings.get_alerting_config()
   )
   print(result["full_report"])  # full Evidently report dict
   ```
3. If drift is real (a sensor recalibration, a process change upstream,
   a genuinely different operating regime): retrain
   (`make train` + `validate_and_promote_model.py`) against recent data
   so the model adapts.
4. If it's a data quality issue (a stuck sensor, a unit conversion bug
   upstream): fix at the source; do NOT silence the alert by raising
   the drift threshold as a first response.

## Rotating the MS Teams webhook URL (Stage 2/3)

Update `TEAMS_WEBHOOK_URL` in `.env` (or the deployment's secret store)
and restart any process that reads `alerting_config.yaml` — no code or
YAML structure change needed, since the URL is interpolated from the
environment at load time.

## Re-downloading the reference dataset

```bash
python scripts/download_data.py --force
```
Only meaningful on a machine with internet (Stage 1). Never run this on
Stage 2/3.

## Building the Stage 3 offline bundle

```bash
bash scripts/build_bundle.sh [output_dir]
```
See `docs/STAGES.md`'s Stage 3 section for the full unpack-and-run
procedure on the target server.

## Common failure: "unable to open database file"

If `MlflowModelRegistry` fails with a sqlite error on a fresh checkout,
the `mlflow/` directory (gitignored, never committed) doesn't exist yet.
This is handled automatically as of the fix in
`pdm.registry.mlflow_registry._ensure_sqlite_parent_dir` — if you see
this error on a version that predates that fix, `mkdir -p mlflow` before
retrying.
