#!/usr/bin/env bash
set -euo pipefail

CONTAINER_NAME="${1:-ha-entity-vault}"
REPO_PATH="${2:-$(pwd)}"
DATA_PATH="${3:-${HOME}/ha-entity-vault-data}"
HOST_PORT="${4:-18000}"
IMAGE="${IMAGE:-ubuntu:24.04}"

if ! command -v lxc >/dev/null 2>&1; then
  echo "lxc command not found. Install LXD/LXC first (Ubuntu: sudo snap install lxd)."
  exit 1
fi

if ! id -nG | tr ' ' '\n' | grep -qx 'lxd'; then
  echo "Current user is not in the 'lxd' group."
  echo "Run: sudo usermod -aG lxd $USER && newgrp lxd"
  exit 1
fi

REPO_PATH="$(realpath "${REPO_PATH}")"
mkdir -p "${DATA_PATH}"
DATA_PATH="$(realpath "${DATA_PATH}")"

if [[ ! -d "${REPO_PATH}/app" ]] || [[ ! -f "${REPO_PATH}/requirements.txt" ]]; then
  echo "REPO_PATH does not look like this project root: ${REPO_PATH}"
  exit 1
fi

lxd init --auto >/dev/null 2>&1 || true

if ! lxc info "${CONTAINER_NAME}" >/dev/null 2>&1; then
  lxc launch "${IMAGE}" "${CONTAINER_NAME}"
else
  echo "Container '${CONTAINER_NAME}' already exists"
fi

if ! lxc config device show "${CONTAINER_NAME}" | grep -q '^app-src:'; then
  lxc config device add "${CONTAINER_NAME}" app-src disk source="${REPO_PATH}" path=/srv/ha-entity-vault shift=true
fi

if ! lxc config device show "${CONTAINER_NAME}" | grep -q '^app-data:'; then
  lxc config device add "${CONTAINER_NAME}" app-data disk source="${DATA_PATH}" path=/data shift=true
fi

if ! lxc config device show "${CONTAINER_NAME}" | grep -q '^web:'; then
  lxc config device add "${CONTAINER_NAME}" web proxy "listen=tcp:0.0.0.0:${HOST_PORT}" connect=tcp:127.0.0.1:8000
fi

lxc restart "${CONTAINER_NAME}"

lxc exec "${CONTAINER_NAME}" -- bash -lc "bash /srv/ha-entity-vault/lxc/setup-in-container.sh"

IP_ADDR="$(lxc list "${CONTAINER_NAME}" -c 4 --format csv | awk -F'[ ,]+' 'NR==1{print $1}')"
echo "Container '${CONTAINER_NAME}' ready."
echo "Container IP: ${IP_ADDR:-unknown}"
echo "Host proxy URL: http://$(hostname -I | awk '{print $1}'):${HOST_PORT}"
