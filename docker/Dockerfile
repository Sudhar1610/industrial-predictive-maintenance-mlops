# General-purpose image: runs the training pipeline, CI helper scripts,
# or an ad-hoc shell. For the leaner API-only image, see Dockerfile.serving.
#
# Multi-stage: the builder stage has a compiler toolchain (needed by a
# couple of compiled dependencies) that the runtime stage never carries,
# keeping the final image smaller and its attack surface lower.
# syntax=docker/dockerfile:1

FROM python:3.11-slim AS builder

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir ".[sql]"

FROM python:3.11-slim AS runtime

RUN groupadd -r pdm && useradd -r -g pdm -d /app pdm
WORKDIR /app

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY src ./src
COPY configs ./configs
COPY scripts ./scripts

ENV PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1

RUN mkdir -p /app/mlflow /app/artifacts /app/logs /app/data/raw && chown -R pdm:pdm /app
USER pdm

ENTRYPOINT ["python"]
CMD ["-m", "pdm.pipelines.training_flow"]
