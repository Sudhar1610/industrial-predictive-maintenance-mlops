# ADR 0003: Local MLflow registry over a managed model registry

## Status

Accepted

## Context

The project needs experiment tracking (params/metrics/artifacts per
training run) and a model registry with stage transitions
(Staging → Production) so a CI gate can promote or block a candidate
model. The two realistic options were: (a) a managed registry (SageMaker
Model Registry, Databricks Unity Catalog, Weights & Biases), or (b) a
self-hosted MLflow instance with a local backend.

Managed options were ruled out immediately by ADR 0001's offline-first
constraint — none of them function without an internet-reachable
service, and the actual Stage 3 production target has no internet at
all. The remaining question was *how* to run MLflow locally: as a
persistent `mlflow server` process (with a database + artifact store
behind it), or as a serverless, client-only setup.

## Decision

Use MLflow's **client-only mode against a local backend**: a
SQLAlchemy-backed tracking URI (`sqlite:///mlflow/mlflow.db`) plus a
local filesystem artifact store (`mlflow/artifacts/`). No
`mlflow server` process is required for tracking, the registry, or
stage transitions — `pdm.registry.mlflow_registry.MlflowModelRegistry`
talks to the sqlite file and local directory directly via
`MlflowClient`.

For the strictest possible offline guarantee at inference time — Stage
3 serving with no dependency on the MLflow client library succeeding at
all, e.g. if the sqlite file is ever unreachable or corrupted —
`pdm.registry.local_artifact` provides a second, independent path that
reads a plain serialized model straight off disk with zero MLflow
involvement. `pdm.serving.model_loader.load_production_model` tries the
registry first and falls back to this path automatically.

## Consequences

**Positive:**
- Full experiment history, param/metric comparison, and Staging→
  Production semantics — the same mental model a team would get from a
  hosted MLflow — with zero infrastructure to keep running. A `docker
  compose` MLflow UI service exists purely as an optional local
  convenience for browsing runs (`docker/docker-compose.yml`'s
  `mlflow-ui`), never a requirement.
- Two independent, working "load the Production model" code paths means
  Stage 3 has a real fallback if the registry is ever unavailable, not
  just a comment saying "consider adding a fallback."
- sqlite + local files are trivially backed up/restored (copy the
  `mlflow/` directory), which matters for a machine with no automated
  ops team watching it.

**Negative / trade-offs accepted:**
- No concurrent-writer safety beyond what sqlite itself provides — this
  is fine for this project's access pattern (one training pipeline run
  at a time, one serving process reading), but would need re-evaluating
  if multiple training jobs ever needed to write to the same registry
  concurrently.
- MLflow's model registry "stages" concept (Staging/Production/Archived)
  is deprecated upstream in favor of aliases/tags as of MLflow 2.9, and
  will eventually be removed. This project uses it anyway because it's
  the clearest mapping to the promotion model this project actually
  wants, and the deprecated API is still fully functional in the pinned
  MLflow version (2.15.1). Migrating to aliases later is a contained
  change inside `pdm.registry.mlflow_registry` only.
- No built-in UI/dashboard without manually running the optional
  `mlflow-ui` compose service — acceptable since this project's serving
  and CI paths never depend on that UI being up.
