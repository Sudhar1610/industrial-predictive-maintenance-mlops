# ADR 0002: A `DataSource` interface, not per-stage branching

## Status

Accepted

## Context

The same application must read training/inference data from three
fundamentally different backends depending on deployment stage: local
CSV files (Stage 1), a real SQL database (Stage 2/3), and MongoDB
(Stage 2/3, for a different plant's telemetry format). The naive
approach — `if stage == "prod": query_sql() else: read_csv()` scattered
through `pdm.features`/`pdm.pipelines` — was rejected before it was
written, because it fails the project's core requirement: promoting
between stages must never touch application code.

## Decision

Define one abstract interface, `pdm.data.base.DataSource`, with exactly
three methods: `fetch_training_data()`, `fetch_latest(unit_id)`, and
`health_check()`. Three concrete implementations —`CsvDataSource`,
`SqlDataSource`, `MongoDataSource` — each satisfy this interface and
return data in the identical tidy shape (one row per `(unit_id, cycle)`,
sensor/setting columns). A factory function,
`pdm.data.factory.get_datasource(config)`, reads `active_source` from
`datasource_config.yaml` and returns the right instance. Every caller —
`pdm.pipelines.training_flow`, `pdm.serving.app`'s `/health` check —
depends only on the `DataSource` type, never a concrete class.

Validation (`pdm.data.validation.validate_sensor_readings`, a Pandera
schema) runs against whatever DataFrame any implementation returns, so
a real SQL/Mongo feed is validated exactly as strictly as the sample CSV
— there's no "trust production data more" special case to accidentally
introduce, which matters because production data is exactly where a
silent schema drift would be most costly.

## Consequences

**Positive:**
- Promoting Stage 1 → 2 is provably a one-line YAML change
  (`active_source: csv` → `sql`), verified by
  `tests/integration/test_config_swap_datasource.py`.
- New backends (e.g. a future OPC-UA or Kafka source) are additive: one
  new class implementing `DataSource`, one new branch in the factory,
  zero changes anywhere else.
- Testing is straightforward: `SqlDataSource`/`MongoDataSource` unit
  tests mock the actual database driver and verify contract behavior
  (query construction, health-check failure handling) without needing a
  live SQL/Mongo server in CI.

**Negative / trade-offs accepted:**
- The interface is deliberately narrow (3 methods). Backend-specific
  operations that don't fit — e.g. `CsvDataSource.fetch_test_data()`
  and `fetch_rul_labels()`, which only make sense for the C-MAPSS
  benchmark's train/test/RUL-label file split — live as extra methods
  on the concrete class, used only by evaluation/dev scripts that
  explicitly need C-MAPSS-specific behavior, never by `pdm.pipelines`
  or `pdm.serving`. This is an accepted asymmetry, not a violation: the
  interface covers what every stage needs in common; anything
  benchmark-specific stays out of it.
- A schema mismatch between SQL/Mongo's real column names and what
  `datasource_config.yaml`'s `csv.column_names`/`feature_engineering.
  sensor_columns` expect is a config problem, not one this abstraction
  prevents — the office/plant team owns getting the real column mapping
  right in `datasource_config.yaml`'s `query_training`/`query_latest`.
