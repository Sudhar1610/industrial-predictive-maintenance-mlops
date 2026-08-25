# Architecture

See the README for the top-level system diagram. This document covers
the design decisions that diagram doesn't show: why the boundaries are
drawn where they are, and how a request/training run actually flows
through the layers.

## Layering rules

```
config  →  data  →  features  →  models  →  evaluation  →  registry
                                                              ↓
                                        pipelines (orchestrates all of the above)
                                                              ↓
                                          serving  ←  alerting  ←  monitoring
```

- **`config`** is read by every other layer but depends on nothing else
  in this project. It's the only place that touches `os.environ` or
  parses YAML directly (`pdm.config.settings`).
- **`data`** depends only on `config` (for connection details) and
  exposes the `DataSource` interface. Nothing downstream of `data`
  imports `CsvDataSource`/`SqlDataSource`/`MongoDataSource` directly —
  always through `pdm.data.factory.get_datasource`.
- **`features`** depends only on `pandas`/`numpy` and `config` (for
  window sizes, sensor column lists). It has no idea where the
  DataFrame it receives came from.
- **`models`** depends only on `numpy`/`config`. The `Model` interface
  (`fit`/`predict`/`save`/`load`/`params`) is the only thing
  `pipelines`, `registry`, and `serving` know about a model.
- **`evaluation`** is pure functions over arrays/DataFrames — no
  dependency on `data`, `models`, or `registry`, so it's trivially unit
  testable and reusable from the CI gate script.
- **`registry`** depends on `models` (to call `.save()`/load via the
  factory) and `config`. `pdm.registry.mlflow_registry` and
  `pdm.registry.local_artifact` are two independent implementations of
  "persist and retrieve a trained Model" — `serving` uses both, in
  order (see below).
- **`pipelines`** is the only layer allowed to import from `data`,
  `features`, `models`, `evaluation`, and `registry` all at once — it's
  the orchestration layer, and its job is exactly to wire the others
  together in sequence.
- **`serving`**, **`alerting`**, **`monitoring`** are the runtime
  triangle: serving loads a model via `registry` and calls
  `features`/`models` the same way `pipelines` does (see
  `pdm.serving.inference` — it reuses `pdm.features.build.build_features`
  directly, not a re-implementation), monitoring logs what serving
  produced and periodically checks it for drift, and a drift breach
  fires through `alerting` — which serving and monitoring both depend on
  but never instantiate a concrete class from.

## Request flow: `/predict`

1. `pdm.serving.app` receives a `PredictRequest` (a unit's recent
   cycle-ordered sensor history).
2. `pdm.serving.inference.build_model_input` turns it into a DataFrame
   (`request_to_dataframe`) and runs it through
   `pdm.features.build.build_features` — **the exact same function
   `pdm.pipelines.training_flow.engineer_features` calls at train time**.
   This is what keeps train/serve feature parity structural: there is
   only one feature-engineering code path, not two that can drift apart.
3. The engineered row (or sequence window, for the LSTM backend) is
   passed to `Model.predict`.
4. The response is logged (`pdm.monitoring.prediction_logger`) and
   recorded to Prometheus (`pdm.monitoring.prometheus_metrics`) before
   being returned.

## Training flow

`pdm.pipelines.training_flow` is a thin Prefect `@flow` sequencing seven
independently-callable `@task` functions (see the module — each one is
short enough to read as documentation of the pipeline shape). The
unit-level train/test split (`split_train_test`) is a specific,
deliberate choice: splitting by row would let cycles from the same
engine appear in both train and test, silently inflating test metrics
because the model has effectively already seen that engine's
degradation curve.

## Model loading: MLflow-first, local-fallback-second

`pdm.serving.model_loader.load_production_model` tries the MLflow
registry first (queries for the `Production`-staged version, downloads
its artifact — which for a local file-store tracking URI is just a
filesystem path, no actual "download") and falls back to
`pdm.registry.local_artifact.load_local_fallback` if that fails for any
reason (no MLflow server/db reachable, no Production version yet, a
corrupted sqlite file). Both code paths read the identical
`Model.save()`/`load()` artifact format, so which one actually served a
given process is invisible to the rest of the app — this is what makes
"MLflow tracking store not deployed on Stage 3" a supported
configuration rather than a special case.

## Why Pandera validation runs identically on all three datasources

`pdm.data.validation.validate_sensor_readings` is called by
`pdm.pipelines.training_flow.validate` against whatever DataFrame
`ingest` produced — it has no idea whether that DataFrame came from a
CSV, a SQL query, or a MongoDB collection scan. A schema violation in
real SQL/Mongo data (a null sensor column, a duplicate `(unit_id,
cycle)` pair from a flaky ingestion job) fails exactly as loudly as a
malformed CSV would in Stage 1 — there is no "trust the real data
source more" special case to accidentally introduce.

See also: [`adr/0002-datasource-abstraction.md`](adr/0002-datasource-abstraction.md)
and [`adr/0003-local-mlflow-over-managed-registry.md`](adr/0003-local-mlflow-over-managed-registry.md)
for the reasoning behind the two biggest architectural bets in this
project.
