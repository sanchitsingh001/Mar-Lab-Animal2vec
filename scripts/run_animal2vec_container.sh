IMAGE="animal2vec:py310"
CONTAINER="animal2vec_run"

# Mount the repo you are CURRENTLY in
HOST_REPO="$(pwd)"

# Optional: where you want results on host (not required for mounting to work)
HOST_RUNS="/cache/a/ssingh/Results/a2v1_results_1"
mkdir -p "${HOST_RUNS}"

CONT_ROOT="/host_root"
CONT_REPO="${CONT_ROOT}/animal2vec"
CONT_CACHE="${CONT_ROOT}/cache"

# If container already exists, remove it so mounts update
docker rm -f "${CONTAINER}" 2>/dev/null || true

docker run -d \
  --name "${CONTAINER}" \
  --gpus all \
  -v "${HOST_REPO}:${CONT_REPO}" \
  -v /cache:"${CONT_CACHE}" \
  -w "${CONT_REPO}" \
  "${IMAGE}" \
  sleep infinity

