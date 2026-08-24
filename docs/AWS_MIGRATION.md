# AWS Migration Map

This project is deliberately local-first (see
[`adr/0001-offline-first-architecture.md`](adr/0001-offline-first-architecture.md))
because Stage 2/3 are air-gapped by requirement. If a future deployment
target *does* have cloud access — a corporate AWS account, not the plant
floor — every local component here has a direct managed-service
equivalent. This table is the migration map for that scenario.

| Local component (this repo) | AWS-managed equivalent | Migration notes |
|---|---|---|
| **MLflow** (sqlite + local files) — experiment tracking, model registry | **SageMaker Model Registry** + **SageMaker Experiments** | `pdm.registry.MlflowModelRegistry` would be swapped for a `SageMakerModelRegistry` implementing the same interface used by `pdm.serving.model_loader`. Model artifacts move from `mlflow/artifacts/` to S3; registry metadata moves from sqlite to SageMaker's managed store. Stage transitions (Staging→Production) map directly to SageMaker Model Package approval status. |
| **Prefect** (local flow engine) | **AWS Step Functions** (simple DAGs) or **MWAA (managed Airflow)** (complex/scheduled DAGs) | The `@task`/`@flow` structure in `training_flow.py` translates 1:1 to Step Functions states or Airflow tasks — each `@task` function's body is already a pure, independently-callable unit. Scheduling (`cron` in Stage 1's dev use) becomes EventBridge rules. |
| **FastAPI** (`pdm.serving.app`, uvicorn) | **SageMaker Endpoint** (managed inference) or **ECS/Fargate + ALB** (if the team wants to keep owning the FastAPI app as-is) | SageMaker Endpoint requires wrapping `Model.predict` in SageMaker's inference container contract (a `model_fn`/`predict_fn` pair) — `pdm.models.base.Model` already has the right shape for this. ECS/Fargate is the lower-effort path: containerize `docker/Dockerfile.serving` as-is and put it behind an Application Load Balancer; no code changes. |
| **Prometheus + Grafana** (`docker-compose.yml`) | **Amazon CloudWatch** (metrics + dashboards) or **Amazon Managed Service for Prometheus (AMP) + Amazon Managed Grafana** | The `/metrics` endpoint already emits standard Prometheus exposition format — AMP can scrape it with zero changes to `pdm.monitoring.prometheus_metrics`. Pure CloudWatch would mean replacing `prometheus_client` calls with `boto3` `put_metric_data` calls in `record_prediction`/`record_drift_breach`. |
| **DVC with a local remote** (data + model-weight versioning) | **DVC with an S3 remote** | This is the smallest possible migration: DVC already supports S3 remotes natively — `dvc remote add -d storage s3://bucket/path` and everything else (the `.dvc` files, the workflow) is unchanged. |
| **GitHub Actions** (`.github/workflows/`) | **AWS CodePipeline** + **CodeBuild** | The job structure (lint → typecheck → test → build → integration-test → model-validation gate) maps to CodeBuild stages inside a CodePipeline. `scripts/validate_and_promote_model.py`'s exit-code contract (0 = proceed, 1 = block) works unchanged as a CodeBuild step. |
| **MS Teams webhook alerter** (`pdm.alerting.teams_alerter`) | **Amazon SNS** (fan-out to email/SMS/Lambda) or keep the Teams webhook and add an **SNS→Lambda→Teams** bridge if centralizing alert routing | A new `SnsAlerter` implementing `pdm.alerting.base.Alerter` is the whole change — `pdm.monitoring.drift_job` and `pdm.serving` never reference a concrete alerter class. |
| **Local SQL/MongoDB `DataSource` implementations** | **Amazon RDS/Aurora** (SQL) or **Amazon DocumentDB** (Mongo-compatible) | No code change at all if the managed service is wire-compatible (DocumentDB is MongoDB-wire-compatible; RDS Postgres/SQL Server work with the existing `SqlDataSource` via SQLAlchemy). Only `datasource_config.yaml`'s `host`/credentials change. |
| **Local file-based prediction log** (`pdm.monitoring.prediction_logger`) | **Amazon Kinesis Data Firehose → S3**, or **CloudWatch Logs** | For high-volume real-time streams, Firehose is the natural fit; `PredictionLogger.log` would be swapped for a Firehose `put_record` call behind the same method signature. |

## Trade-offs of moving to AWS

**Gains:**
- No self-managed uptime/patching for MLflow, Prometheus, Grafana.
- Elastic compute for training (SageMaker Training Jobs) instead of a
  fixed local machine.
- IAM-based access control instead of `.env` file secrets.
- Built-in HA/multi-AZ for the registry and data stores.

**Costs:**
- Every managed service is billed continuously (SageMaker Endpoints
  bill per-hour even at zero traffic; RDS/DocumentDB likewise).
- Loses the "zero services, zero installs" offline guarantee this
  project is built around — not a viable path for the actual Stage 3
  IIOT server, which by definition has no AWS connectivity.
- Adds real operational complexity (IAM policies, VPC networking,
  service quotas) that this project's local-first design avoids
  entirely for a portfolio/demo audience.

**Bottom line**: this migration map exists for the case where a *future,
different* deployment target has cloud access (e.g., a corporate BI
environment, not the plant floor). The actual production target for
this specific project — the air-gapped IIOT server — will never take
this path, which is exactly why the local-first architecture was chosen
in the first place. See
[`adr/0001-offline-first-architecture.md`](adr/0001-offline-first-architecture.md).
