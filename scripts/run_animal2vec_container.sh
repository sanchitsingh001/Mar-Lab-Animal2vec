#!/usr/bin/env bash
set -euo pipefail

IMAGE="animal2vec:py310"
CONTAINER="animal2vec_run"

# ---- Host paths (cssc) ----
HOST_REPO="/home/ssingh/Mar-Lab-Animal2vec"

# Prefer the big NVMe data mount (recommended)
HOST_DATA_ROOT="/storage/data2/rochlab-data/ssingh"
HOST_DATASET="${HOST_DATA_ROOT}/Datasets/Xeno-canto"

# Where you want results/checkpoints/logs on host
HOST_RUNS="${HOST_DATA_ROOT}/Results/animal2vec_runs"
mkdir -p "${HOST_RUNS}"

# ---- Container paths ----
CONT_ROOT="/host"
CONT_REPO="${CONT_ROOT}/repo"
CONT_DATA="${CONT_ROOT}/data"
CONT_RUNS="${CONT_ROOT}/runs"

# If container already exists, remove it so mounts update
docker rm -f "${CONTAINER}" 2>/dev/null || true

# Run detached and keep alive
docker run -d \
  --name "${CONTAINER}" \
  --gpus all \
  -v "${HOST_REPO}:${CONT_REPO}" \
  -v "${HOST_DATASET}:${CONT_DATA}:ro" \
  -v "${HOST_RUNS}:${CONT_RUNS}" \
  -w "${CONT_REPO}" \
  "${IMAGE}" \
  sleep infinity

echo "Container started: ${CONTAINER}"
echo "Repo mounted at:   ${CONT_REPO}"
echo "Dataset mounted at:${CONT_DATA} (read-only)"
echo "Runs mounted at:   ${CONT_RUNS}"

