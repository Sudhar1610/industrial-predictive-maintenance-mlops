"""Central config loader.

Reads the three YAML files under `configs/`, interpolates `${ENV_VAR}`
placeholders against the process environment (via python-dotenv for local
`.env` files, falling back to real env vars in Stage 2/3), validates the
result against the schemas in `schemas.py`, and exposes cached singletons.

This module is the ONLY place that should read YAML files or `os.environ`
directly. Every other module receives config through these functions (or
through dependency injection in constructors), so config lookups stay
testable and centralized.
"""

from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from pdm.config.schemas import (
    AlertingConfig,
    DatasourceConfig,
    LoggingConfig,
    ModelConfig,
)

_ENV_VAR_PATTERN = re.compile(r"\$\{([A-Z0-9_]+)\}")

# Resolved once at import time; overridable via PDM_CONFIG_DIR for tests
# or for pointing at a different config directory entirely.
CONFIG_DIR = Path(os.environ.get("PDM_CONFIG_DIR", "configs"))

load_dotenv(override=False)


def _interpolate_env_vars(value: Any) -> Any:
    """Recursively replace `${VAR_NAME}` placeholders with environment
    values. A referenced-but-unset variable is left as an empty string
    rather than raising, so that non-active backends (e.g. `sql:` while
    running Stage 1 on `csv`) don't block startup for missing secrets
    they'll never use."""
    if isinstance(value, str):
        return _ENV_VAR_PATTERN.sub(lambda m: os.environ.get(m.group(1), ""), value)
    if isinstance(value, dict):
        return {k: _interpolate_env_vars(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_interpolate_env_vars(v) for v in value]
    return value


def _load_yaml(filename: str) -> dict[str, Any]:
    path = CONFIG_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found: {path}. "
            f"Set PDM_CONFIG_DIR to override the config directory."
        )
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return _interpolate_env_vars(raw)


@lru_cache(maxsize=1)
def get_datasource_config() -> DatasourceConfig:
    """Load and validate `datasource_config.yaml`."""
    return DatasourceConfig.model_validate(_load_yaml("datasource_config.yaml"))


@lru_cache(maxsize=1)
def get_alerting_config() -> AlertingConfig:
    """Load and validate `alerting_config.yaml`."""
    return AlertingConfig.model_validate(_load_yaml("alerting_config.yaml"))


@lru_cache(maxsize=1)
def get_model_config() -> ModelConfig:
    """Load and validate `model_config.yaml`."""
    return ModelConfig.model_validate(_load_yaml("model_config.yaml"))


@lru_cache(maxsize=1)
def get_logging_config() -> LoggingConfig:
    """Load and validate `logging_config.yaml`."""
    return LoggingConfig.model_validate(_load_yaml("logging_config.yaml"))


def clear_config_cache() -> None:
    """Drop all cached, parsed configs so the next `get_*_config()` call
    re-reads YAML from disk. Used by tests that swap `PDM_CONFIG_DIR` or
    edit a config file mid-test to prove a config-only stage switch."""
    get_datasource_config.cache_clear()
    get_alerting_config.cache_clear()
    get_model_config.cache_clear()
    get_logging_config.cache_clear()
