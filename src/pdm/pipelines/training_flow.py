"""Prefect flow: ingest -> validate -> feature -> train -> evaluate -> register.

Running `python -m pdm.pipelines.training_flow` (or `make train`)
executes this deterministically end-to-end against whichever datasource
`configs/datasource_config.yaml` currently points at -- CSV in Stage 1,
real SQL/Mongo in Stage 2/3 -- with zero code changes required to
retarget it. Each stage is a plain, independently-testable function
wrapped in a Prefect `@task`; the `@flow` just sequences them, so this
module stays readable as documentation of the pipeline shape even
without Prefect's orchestration UI running.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd
from loguru import logger
from prefect import flow, task

from pdm.config import settings
from pdm.config.schemas import DatasourceConfig, ModelConfig
from pdm.data.factory import get_datasource
from pdm.data.validation import validate_sensor_readings
from pdm.evaluation.metrics import classification_metrics, regression_metrics
from pdm.features.build import build_features, get_feature_columns
from pdm.features.labels import add_rul_and_failure_labels
from pdm.features.sequences import build_sequences
from pdm.models.base import Model
from pdm.models.factory import get_model
from pdm.registry.mlflow_registry import MlflowModelRegistry


def _to_model_arrays(
    df: pd.DataFrame, model_config: ModelConfig
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int_], npt.NDArray[np.float64]]:
    """Shape an engineered+labeled DataFrame into `(X, y_class, y_reg)`
    for whichever backend is active -- 3D sequences for `lstm`, a flat
    2D matrix for the tabular backends."""
    feature_columns = get_feature_columns(df)
    if model_config.active_model == "lstm":
        return build_sequences(df, feature_columns, model_config.lstm.sequence_length)
    X = df[feature_columns].to_numpy(dtype=np.float64)
    return X, df["will_fail"].to_numpy(), df["RUL"].to_numpy(dtype=np.float64)


@task(name="ingest", retries=1)
def ingest(datasource_config: DatasourceConfig) -> pd.DataFrame:
    source = get_datasource(datasource_config)
    if not source.health_check():
        raise RuntimeError("Datasource health check failed; aborting ingest.")
    df = source.fetch_training_data()
    logger.info("Ingested {} rows, {} units.", len(df), df["unit_id"].nunique())
    return df


@task(name="validate")
def validate(df: pd.DataFrame, datasource_config: DatasourceConfig) -> pd.DataFrame:
    return validate_sensor_readings(
        df, fail_on_violation=datasource_config.validation.fail_on_violation
    )


@task(name="engineer_features")
def engineer_features(df: pd.DataFrame, model_config: ModelConfig) -> pd.DataFrame:
    engineered = build_features(df, model_config.feature_engineering)
    return add_rul_and_failure_labels(
        engineered,
        failure_horizon_cycles=model_config.classification.failure_horizon_cycles,
        rul_cap=model_config.regression.rul_cap,
    )


@task(name="split_train_test")
def split_train_test(
    df: pd.DataFrame, test_split: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Unit-level split: every cycle of a given engine stays entirely in
    train or entirely in test, so the test metrics reflect generalizing
    to unseen engines rather than unseen cycles of an already-seen one.
    """
    unit_ids = df["unit_id"].unique()
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unit_ids)
    n_test = max(1, int(len(shuffled) * test_split))
    test_units = set(shuffled[:n_test])

    train_df = df[~df["unit_id"].isin(test_units)].reset_index(drop=True)
    test_df = df[df["unit_id"].isin(test_units)].reset_index(drop=True)
    logger.info(
        "Split: {} train units, {} test units.", len(unit_ids) - len(test_units), len(test_units)
    )
    return train_df, test_df


@task(name="train_model")
def train_model(train_df: pd.DataFrame, model_config: ModelConfig) -> Model:
    X, y_class, y_reg = _to_model_arrays(train_df, model_config)
    model = get_model(model_config)
    model.fit(X, y_class, y_reg)
    logger.info(
        "Trained {} on {} samples ({} feature dims).", model_config.active_model, len(X), X.shape[-1]
    )
    return model


@task(name="evaluate_model")
def evaluate_model(model: Model, test_df: pd.DataFrame, model_config: ModelConfig) -> dict[str, float]:
    X, y_class, y_reg = _to_model_arrays(test_df, model_config)
    result = model.predict(X)
    metrics = {
        **classification_metrics(y_class, result.will_fail, result.failure_probability),
        **regression_metrics(y_reg, result.remaining_useful_life),
    }
    logger.info("Evaluation metrics: {}", metrics)
    return metrics


@task(name="register_model")
def register_model(model: Model, metrics: dict[str, float], model_config: ModelConfig) -> str:
    """Register the newly-trained model as a new version in Staging.
    Promotion to Production is a separate, deliberate step -- gated by
    CI's model-validation check (`scripts/validate_and_promote_model.py`)
    so a regression never reaches Production automatically."""
    registry = MlflowModelRegistry(model_config.registry)
    version = registry.log_training_run(model, params=model.params, metrics=metrics)
    registry.transition_stage(version, "Staging")
    logger.info("Registered {} version {} (Staging).", model_config.registry.model_name, version)
    return version


@flow(name="pdm-training-pipeline", log_prints=True)
def training_flow() -> str:
    """Ingest -> validate -> feature-engineer -> split -> train ->
    evaluate -> register. Returns the newly-registered model version.
    """
    datasource_config = settings.get_datasource_config()
    model_config = settings.get_model_config()

    raw = ingest(datasource_config)
    validated = validate(raw, datasource_config)
    labeled = engineer_features(validated, model_config)
    train_df, test_df = split_train_test(
        labeled, model_config.training.test_split, model_config.training.seed
    )

    model = train_model(train_df, model_config)
    metrics = evaluate_model(model, test_df, model_config)
    return register_model(model, metrics, model_config)


if __name__ == "__main__":
    training_flow()
