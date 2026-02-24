#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if [[ ! -d "${REPO_ROOT}/deploy/autotest/node_modules/playwright" ]]; then
  echo "[harness] Missing deploy/autotest/node_modules/playwright."
  echo "[harness] Runtime install fallback is disabled."
  echo "[harness] Use the baked runner image from deploy/autotest/runner.Dockerfile,"
  echo "[harness] or run 'cd deploy/autotest && npm install' before invoking this wrapper."
  exit 1
fi

exec node "${REPO_ROOT}/deploy/autotest/scripts/assert-config-timestamps.mjs"
