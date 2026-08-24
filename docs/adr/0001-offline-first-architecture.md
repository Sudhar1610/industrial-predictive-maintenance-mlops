# ADR 0001: Offline-first architecture

## Status

Accepted

## Context

The real production target for this project is an industrial IIOT
server with **no internet access at all**, running on an air-gapped
plant network. Before that, the project passes through an office PWS
that is also air-gapped (no `pip`, no external package registries, no
SaaS APIs) and only reachable for internal SQL/MongoDB and MS Teams
traffic.

A large fraction of the standard MLOps toolchain assumes the opposite:
MLflow is commonly deployed as a hosted server; Prefect has a paid Cloud
offering promoted heavily in its docs; monitoring stacks default to
SaaS dashboards (Datadog, Weights & Biases); most Python projects assume
`pip install` works at deploy time.

## Decision

Every component in this project must run with **zero runtime internet
access and zero runtime service dependencies beyond what we explicitly
choose to run locally**:

- MLflow: local sqlite + local filesystem artifact store, no
  `mlflow server` process required (though one may optionally be run
  for the local dev UI — see `docker-compose.yml`'s `mlflow-ui`
  service, which is convenience-only, not load-bearing).
- Prefect: the open-source local flow engine, no Prefect Cloud.
- Evidently, Prometheus, Grafana: all self-hosted, local config files,
  no SaaS accounts.
- Package installation: conda-only on Stage 2/3, with the one
  unavoidable pip-installed exception (`pandera`, `evidently` — not
  reliably on conda-forge) installed once during environment build on a
  machine with internet, then frozen into a `conda-pack` tarball that
  needs no further installs anywhere it's unpacked.
- Data download: exactly one script (`scripts/download_data.py`) is
  allowed to touch the network, and it never runs on Stage 2/3.

## Consequences

**Positive:**
- The same code and config structure that runs on a laptop during
  development is provably deployable to the actual production target,
  because it was built against that target's constraints from day one
  rather than retrofitted later.
- No hidden "works on my machine, breaks in the plant" surprises from a
  library that silently phones home (telemetry pings, license checks,
  update checks).
- Forces config-driven design (see ADR 0002) as a side effect: since
  nothing can assume a specific network-reachable service, everything
  has to be parameterized and swappable.

**Negative / trade-offs accepted:**
- Loses the convenience of managed services: no automatic backups,
  scaling, or high availability for the local MLflow/Prometheus/Grafana
  stack — these are genuinely single-machine, single-point-of-failure
  setups. Acceptable because Stage 3's actual requirement is "runs
  reliably on one dedicated industrial server," not "scales to
  multi-tenant cloud load."
- A few dependencies (`pandera`, `evidently`) require one pip install at
  environment-build time rather than being pure-conda; this is an
  explicit, documented exception (see `environment.yml`), not a silent
  gap.
- If cloud connectivity ever *does* become available for a different
  deployment target, this project doesn't get managed-service benefits
  for free — see `docs/AWS_MIGRATION.md` for what that migration would
  look like.
