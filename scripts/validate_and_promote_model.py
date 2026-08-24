#!/usr/bin/env python3
"""CI model-validation gate.

Promotes the latest Staging model version to Production ONLY IF it does
not regress the configured guardrail metric vs the current Production
model (or promotes unconditionally if there is no Production model yet).

Exit code 0 = promoted (or nothing to do). Exit code 1 = BLOCKED: the
candidate regresses. `.github/workflows/model_validation.yml` runs this
after training; a non-zero exit fails that CI job and Production stays
on the previous model -- this is what "blocks promotion if metrics
regress" means in practice.

Usage:
    python scripts/validate_and_promote_model.py [--tolerance 0.02]
"""

from __future__ import annotations

import argparse
import sys

from loguru import logger

from pdm.config import settings
from pdm.registry.mlflow_registry import MlflowModelRegistry

# Metrics where higher is better vs where lower is better -- needed to
# know which direction counts as "regression."
_HIGHER_IS_BETTER = {"f1", "roc_auc", "accuracy", "precision", "recall", "r2"}
_LOWER_IS_BETTER = {"rmse", "mae", "nasa_score"}


def _regressed(metric_name: str, candidate: float, baseline: float, tolerance: float) -> bool:
    if metric_name in _HIGHER_IS_BETTER:
        return candidate < baseline * (1 - tolerance)
    if metric_name in _LOWER_IS_BETTER:
        return candidate > baseline * (1 + tolerance)
    raise ValueError(
        f"Unknown guardrail metric {metric_name!r}; add it to _HIGHER_IS_BETTER or _LOWER_IS_BETTER."
    )


def validate_and_promote(tolerance: float = 0.02) -> int:
    model_config = settings.get_model_config()
    registry = MlflowModelRegistry(model_config.registry)

    staging_versions = registry._client.get_latest_versions(
        model_config.registry.model_name, stages=["Staging"]
    )
    if not staging_versions:
        logger.info("No Staging version to evaluate; nothing to do.")
        return 0

    candidate_version = str(staging_versions[0].version)
    candidate_metrics = registry.get_version_metrics(candidate_version)
    guardrail_metric = model_config.classification.target_metric

    production = registry.get_production_version()
    if production is None:
        logger.info("No existing Production version; promoting version {}.", candidate_version)
        registry.transition_stage(candidate_version, "Production")
        return 0

    baseline_metrics = registry.get_version_metrics(str(production.version))
    baseline_value = baseline_metrics.get(guardrail_metric)
    candidate_value = candidate_metrics.get(guardrail_metric)

    if baseline_value is None or candidate_value is None:
        logger.error(
            "Guardrail metric '{}' missing from candidate or Production run; blocking promotion.",
            guardrail_metric,
        )
        return 1

    if _regressed(guardrail_metric, candidate_value, baseline_value, tolerance):
        logger.error(
            "BLOCKED: candidate {}={:.4f} regresses vs Production {}={:.4f} (version {}).",
            guardrail_metric,
            candidate_value,
            guardrail_metric,
            baseline_value,
            production.version,
        )
        return 1

    logger.info(
        "PROMOTING version {}: {}={:.4f} (Production was {:.4f}).",
        candidate_version,
        guardrail_metric,
        candidate_value,
        baseline_value,
    )
    registry.transition_stage(candidate_version, "Production")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.02,
        help="Fractional regression tolerance before blocking (default 0.02 = 2%%).",
    )
    args = parser.parse_args()
    sys.exit(validate_and_promote(tolerance=args.tolerance))


if __name__ == "__main__":
    main()
