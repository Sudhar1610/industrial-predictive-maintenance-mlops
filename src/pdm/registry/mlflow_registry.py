"""Local MLflow experiment tracking + model registry.

Uses a SQLAlchemy-backed tracking URI (`sqlite:///mlflow/mlflow.db` by
default) with a local filesystem artifact store -- no `mlflow server`
process is required for either tracking or the model registry; the
`MlflowClient` talks to the sqlite file and local directory directly.
This is what "local registry" means throughout this project: durable,
queryable model history without a service to keep running.

For the fully-offline Stage 3 inference path where even the MLflow
client library should not be load-bearing, see `pdm.registry.local_artifact`
instead -- both paths read/write the same `Model.save()`/`load()` artifact
format, so a model promoted here is also loadable there.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import mlflow
from loguru import logger
from mlflow.entities.model_registry import ModelVersion
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from pdm.config.schemas import RegistryConfig
from pdm.models.base import Model
from pdm.models.factory import load_model

_EXPERIMENT_NAME = "pdm_training"


class MlflowModelRegistry:
    """Thin wrapper around `MlflowClient` scoped to this project's
    registered model and experiment."""

    def __init__(self, config: RegistryConfig) -> None:
        self._config = config
        mlflow.set_tracking_uri(config.tracking_uri)
        self._client = MlflowClient(tracking_uri=config.tracking_uri)
        self._experiment_id = self._ensure_experiment()
        self._ensure_registered_model()

    def _ensure_experiment(self) -> str:
        experiment = self._client.get_experiment_by_name(_EXPERIMENT_NAME)
        if experiment is not None:
            return experiment.experiment_id
        artifact_location = str(Path(self._config.artifact_location).resolve())
        Path(artifact_location).mkdir(parents=True, exist_ok=True)
        return self._client.create_experiment(_EXPERIMENT_NAME, artifact_location=artifact_location)

    def _ensure_registered_model(self) -> None:
        try:
            self._client.get_registered_model(self._config.model_name)
        except MlflowException:
            self._client.create_registered_model(self._config.model_name)
            logger.info("Created new registered model: {}", self._config.model_name)

    def log_training_run(
        self, model: Model, params: dict[str, Any], metrics: dict[str, float]
    ) -> str:
        """Log a completed training run's params/metrics, save the
        model's artifact directory into the run, and register it as a
        new (unstaged) model version. Returns the new version number.
        """
        with mlflow.start_run(experiment_id=self._experiment_id) as run:
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            with tempfile.TemporaryDirectory() as tmp_dir:
                model_dir = Path(tmp_dir) / "model"
                model.save(model_dir)
                mlflow.log_artifacts(str(model_dir), artifact_path="model")
            run_id = run.info.run_id

        model_version = self._client.create_model_version(
            name=self._config.model_name,
            source=f"runs:/{run_id}/model",
            run_id=run_id,
        )
        logger.info(
            "Registered {} version {} (run_id={})",
            self._config.model_name,
            model_version.version,
            run_id,
        )
        return str(model_version.version)

    def transition_stage(self, version: str, stage: str) -> None:
        """Move `version` to `stage` (one of "Staging", "Production",
        "Archived", "None"). Promoting to Production automatically
        archives any version currently in Production, so there is always
        at most one Production version.

        Uses MLflow's classic stages API (deprecated upstream in favor of
        aliases/tags as of 2.9, but still functional and the clearest
        mapping to this project's Staging->Production promotion model).
        """
        self._client.transition_model_version_stage(
            name=self._config.model_name,
            version=version,
            stage=stage,
            archive_existing_versions=(stage == "Production"),
        )
        logger.info("Transitioned {} version {} -> {}", self._config.model_name, version, stage)

    def get_production_version(self) -> ModelVersion | None:
        versions = self._client.get_latest_versions(self._config.model_name, stages=["Production"])
        return versions[0] if versions else None

    def get_version_metrics(self, version: str) -> dict[str, float]:
        model_version = self._client.get_model_version(self._config.model_name, version)
        run = self._client.get_run(model_version.run_id)
        return dict(run.data.metrics)

    def load_production_model(self) -> Model:
        """Load the current Production model via the MLflow registry.
        Requires the sqlite tracking store and local artifact directory
        to be reachable -- for the harder offline guarantee (no MLflow
        dependency at all at inference time), use
        `pdm.registry.local_artifact.load_local_fallback` instead."""
        production = self.get_production_version()
        if production is None:
            raise RuntimeError(
                f"No Production version registered for '{self._config.model_name}'."
            )
        local_path = mlflow.artifacts.download_artifacts(
            artifact_uri=f"models:/{self._config.model_name}/Production"
        )
        return load_model(Path(local_path))
