#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
OV_TRAIN_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"
PROXY_ENV="${VAKA_VIKINGBOT_PROXY_ENV:-${OV_TRAIN_ROOT}/evolution/components/ov-benchmark-proxy/.env}"

if [[ -f "${PROXY_ENV}" ]]; then
  set -a
  # Reuse evaluator/TOS identities without duplicating secrets in this adapter.
  source "${PROXY_ENV}"
  set +a
fi

export PYTHONPATH="${REPO_ROOT}/bot:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export VAKA_VIKINGBOT_CASE_SERVICE_URL="${VAKA_VIKINGBOT_CASE_SERVICE_URL:-http://127.0.0.1:8765}"

HOST="${VAKA_VIKINGBOT_HOST:-127.0.0.1}"
PORT="${VAKA_VIKINGBOT_PORT:-8766}"

cd "${REPO_ROOT}"
exec .venv/bin/python -m benchmark.vaka.vikignbot.service_app \
  --host "${HOST}" \
  --port "${PORT}" \
  --log-level "${VAKA_VIKINGBOT_LOG_LEVEL:-info}"
