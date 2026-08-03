#!/usr/bin/env bash
set -euo pipefail

# Tau2 convenience launcher for the generic OpenViking session/train batch pipeline.
# Start the tau2 runtime service first:
#   bash benchmark/tau2/train/run_service.sh --host 127.0.0.1 --port 1944
# Pass --rollout-backend native|vikingbot to override per run (default: vikingbot).
# Pass --no-eval-each-epoch to disable the Tau2 default per-epoch eval while
# keeping the final eval enabled unless --skip-final-eval is also provided.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TAU2_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${TAU2_DIR}/../.." && pwd)"

HAS_BENCHMARK_SERVICE_URL=0
for arg in "$@"; do
  case "${arg}" in
    --benchmark-service-url|--benchmark-service-url=*)
      HAS_BENCHMARK_SERVICE_URL=1
      ;;
  esac
done
if [[ "${HAS_BENCHMARK_SERVICE_URL}" == "0" ]]; then
  set -- --benchmark-service-url "http://127.0.0.1:1944" "$@"
fi

exec "${REPO_ROOT}/openviking/session/train/run_batch_train_eval.sh" \
  --dataset tau2 \
  --domain airline \
  --eval-each-epoch \
  --concurrency 50 \
  --commit-concurrency 150 \
  "$@"
