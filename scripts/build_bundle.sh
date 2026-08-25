#!/usr/bin/env bash
# Packages the exact conda environment (via conda-pack) plus this
# project's source into a single relocatable tarball for handoff to
# Stage 3 (the fully offline IIOT server). Run this on a machine WITH
# internet access (Stage 1 dev box, or the office PWS during Stage 2) --
# it is the packaging step, not something that runs ON the offline server.
#
# What conda-pack solves: a plain `conda env export` + `conda env create`
# on the target machine would need internet to re-resolve and download
# packages. conda-pack instead freezes the ALREADY-RESOLVED environment
# (every binary, every dependency) into a tarball that unpacks and runs
# with zero network access and zero package installs -- exactly what
# Stage 3 requires.
#
# Usage:
#   bash scripts/build_bundle.sh [output_dir]
#
# Produces:
#   <output_dir>/pdm-env.tar.gz       (the packed conda environment)
#   <output_dir>/pdm-bundle.tar.gz    (env + src/configs/scripts, ready to ship)

set -euo pipefail

ENV_NAME="pdm"
OUTPUT_DIR="${1:-bundle}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "==> Building offline deployment bundle for Stage 3"
echo "    Project root: ${PROJECT_ROOT}"
echo "    Output dir:   ${OUTPUT_DIR}"

if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda is required to build the bundle (this script packages a" >&2
    echo "conda environment). Install Miniconda/Miniforge first." >&2
    exit 1
fi

if ! conda env list | grep -q "^${ENV_NAME} "; then
    echo "==> Environment '${ENV_NAME}' not found; creating it from environment.yml"
    conda env create -f "${PROJECT_ROOT}/environment.yml"
else
    echo "==> Updating existing '${ENV_NAME}' environment from environment.yml"
    conda env update -f "${PROJECT_ROOT}/environment.yml"
fi

if ! conda run -n "${ENV_NAME}" python -c "import conda_pack" >/dev/null 2>&1; then
    echo "==> Installing conda-pack into '${ENV_NAME}'"
    conda install -n "${ENV_NAME}" -y -c conda-forge conda-pack
fi

mkdir -p "${OUTPUT_DIR}"

echo "==> Packing environment '${ENV_NAME}' (this can take several minutes)"
conda run -n base conda-pack -n "${ENV_NAME}" -o "${OUTPUT_DIR}/pdm-env.tar.gz" --force

echo "==> Assembling full bundle (env + source + configs + scripts)"
BUNDLE_STAGING="$(mktemp -d)"
trap 'rm -rf "${BUNDLE_STAGING}"' EXIT

mkdir -p "${BUNDLE_STAGING}/pdm-bundle/env"
tar -xzf "${OUTPUT_DIR}/pdm-env.tar.gz" -C "${BUNDLE_STAGING}/pdm-bundle/env"

cp -r "${PROJECT_ROOT}/src" "${BUNDLE_STAGING}/pdm-bundle/src"
cp -r "${PROJECT_ROOT}/configs" "${BUNDLE_STAGING}/pdm-bundle/configs"
cp -r "${PROJECT_ROOT}/scripts" "${BUNDLE_STAGING}/pdm-bundle/scripts"
cp "${PROJECT_ROOT}/pyproject.toml" "${BUNDLE_STAGING}/pdm-bundle/"
mkdir -p "${BUNDLE_STAGING}/pdm-bundle/artifacts" "${BUNDLE_STAGING}/pdm-bundle/mlflow"

cat > "${BUNDLE_STAGING}/pdm-bundle/UNPACK_AND_RUN.md" <<'EOF'
# Stage 3 unpack-and-run procedure

This bundle contains a fully self-sufficient, pre-resolved conda
environment plus this project's source. No internet access, and no
`conda`/`pip` install of any kind, is required on the target server.

1. Copy `pdm-bundle.tar.gz` to the IIOT server by whatever offline
   transfer method is approved there (USB, internal file share, etc.).
2. Extract it:
   ```
   tar -xzf pdm-bundle.tar.gz
   cd pdm-bundle
   ```
3. Activate the packed environment (conda-pack's own activation, not a
   real `conda activate` -- no conda installation is needed on this
   machine at all):
   ```
   source env/bin/activate
   conda-unpack   # fixes up hardcoded paths baked in at pack time; run once
   ```
4. Point the app at this bundle's config and source:
   ```
   export PDM_CONFIG_DIR="$(pwd)/configs"
   export PYTHONPATH="$(pwd)/src"
   ```
5. Edit `configs/datasource_config.yaml` (`active_source: sql` or
   `mongodb`, real connection details via env vars) and
   `configs/alerting_config.yaml` (`active_channel: teams_webhook`) for
   this server's real data sources and alert channel -- see
   `docs/STAGES.md`. This is the ONLY change needed; no source files
   are edited.
6. Copy over (or re-train fresh against real data, if that's the
   procedure for this deployment) a Production model artifact into
   `artifacts/production_model/` or `mlflow/`, matching
   `configs/model_config.yaml`'s `registry` section.
7. Run the serving API with no server process required beyond this one:
   ```
   python -m uvicorn pdm.serving.app:app --host 0.0.0.0 --port 8000
   ```
   Or run the training/scoring pipeline directly:
   ```
   python -m pdm.pipelines.training_flow
   ```

Deactivate with `source env/bin/deactivate` when done.
EOF

echo "==> Creating final bundle tarball"
tar -czf "${OUTPUT_DIR}/pdm-bundle.tar.gz" -C "${BUNDLE_STAGING}" pdm-bundle

echo "==> Done."
echo "    Packed env only:  ${OUTPUT_DIR}/pdm-env.tar.gz"
echo "    Full bundle:      ${OUTPUT_DIR}/pdm-bundle.tar.gz"
echo "    See UNPACK_AND_RUN.md inside the bundle for the Stage 3 deployment procedure."
