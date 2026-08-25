"""Pydantic models describing the shape of each YAML config file.

These models are the single source of truth for what a valid config looks
like. Loading a YAML that doesn't match one of these raises a validation
error immediately at startup (fail loud), rather than surfacing as a
confusing runtime error three layers down in a pipeline.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

# --------------------------------------------------------------------------
# datasource_config.yaml
# --------------------------------------------------------------------------


class CsvSourceConfig(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    train_path: str
    test_path: str
    rul_path: str
    column_names: list[str]


class SqlSourceConfig(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    driver: str
    host: str
    port: int
    database: str
    username: str
    password: str
    odbc_driver: str | None = None
    query_training: str
    query_latest: str
    connection_timeout_s: int = 10


class MongoSourceConfig(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    uri: str
    database: str
    collection_training: str
    collection_latest: str
    connection_timeout_ms: int = 5000


class DataValidationConfig(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    fail_on_violation: bool = True
    min_rows_expected: int = 1


class DatasourceConfig(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    active_source: Literal["csv", "sql", "mongodb"]
    csv: CsvSourceConfig
    sql: SqlSourceConfig
    mongodb: MongoSourceConfig
    validation: DataValidationConfig


# --------------------------------------------------------------------------
# alerting_config.yaml
# --------------------------------------------------------------------------


class ConsoleAlertConfig(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    min_severity: Literal["info", "warning", "critical"] = "info"


class TeamsWebhookAlertConfig(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    webhook_url: str
    min_severity: Literal["info", "warning", "critical"] = "warning"
    card_theme_color: str = "FF0000"
    timeout_s: int = 5
    retry_attempts: int = 2


class AlertTriggerConfig(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    drift_breach: bool = True
    failure_probability_threshold: float = 0.8
    rul_below_cycles: int = 15
    model_health_check_failure: bool = True


class AlertingConfig(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    active_channel: Literal["console", "teams_webhook"]
    console: ConsoleAlertConfig
    teams_webhook: TeamsWebhookAlertConfig
    triggers: AlertTriggerConfig


# --------------------------------------------------------------------------
# model_config.yaml
# --------------------------------------------------------------------------


class FeatureEngineeringConfig(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    rolling_windows: list[int]
    lag_steps: list[int]
    degradation_slope_window: int
    sensor_columns: list[str]


class ClassificationConfig(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    failure_horizon_cycles: int
    target_metric: str = "f1"
    decision_threshold: float = 0.5


class RegressionConfig(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    target_metric: str = "rmse"
    rul_cap: int = 130


class XgboostConfig(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    n_estimators: int = 300
    max_depth: int = 6
    learning_rate: float = 0.05
    random_state: int = 42


class SklearnRfConfig(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    n_estimators: int = 200
    max_depth: int = 12
    random_state: int = 42


class LstmConfig(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    sequence_length: int = 30
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.2
    epochs: int = 50
    batch_size: int = 64
    learning_rate: float = 0.001
    random_state: int = 42


class TrainingConfig(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    test_split: float = 0.2
    seed: int = 42


class RegistryConfig(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    tracking_uri: str
    artifact_location: str
    model_name: str
    local_fallback_path: str


class ModelConfig(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    active_model: Literal["sklearn_rf", "xgboost", "lstm"]
    feature_engineering: FeatureEngineeringConfig
    classification: ClassificationConfig
    regression: RegressionConfig
    xgboost: XgboostConfig
    sklearn_rf: SklearnRfConfig
    lstm: LstmConfig
    training: TrainingConfig
    registry: RegistryConfig


# --------------------------------------------------------------------------
# logging_config.yaml
# --------------------------------------------------------------------------


class LoggingConfig(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    level: str = "INFO"
    sink: str = "stdout"
    rotation: str = "50 MB"
    retention: str = "14 days"
    serialize: bool = False


# Ordering used to compare a firing alert's severity against a channel's
# `min_severity` threshold (e.g. "warning" >= "info" fires, "info" against
# a "warning" threshold does not).
SEVERITY_ORDER: dict[str, int] = {"info": 0, "warning": 1, "critical": 2}
