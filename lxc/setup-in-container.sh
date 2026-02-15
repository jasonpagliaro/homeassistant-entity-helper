#!/usr/bin/env bash
set -euo pipefail

APP_USER="haev"
APP_GROUP="haev"
APP_DIR="/srv/ha-entity-vault"
VENV_DIR="/opt/ha-entity-vault/.venv"
ENV_FILE="/etc/default/ha-entity-vault"
SERVICE_FILE="/etc/systemd/system/ha-entity-vault.service"

if [[ "${EUID}" -ne 0 ]]; then
  echo "Run as root inside the LXC container."
  exit 1
fi

if [[ ! -f "${APP_DIR}/requirements.txt" ]]; then
  echo "Missing ${APP_DIR}/requirements.txt. Ensure project is mounted into the container."
  exit 1
fi

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y python3 python3-venv ca-certificates curl

if ! id -u "${APP_USER}" >/dev/null 2>&1; then
  useradd --system --create-home --shell /usr/sbin/nologin "${APP_USER}"
fi

mkdir -p /opt/ha-entity-vault /data
chown -R "${APP_USER}:${APP_GROUP}" /opt/ha-entity-vault
if ! chown -R "${APP_USER}:${APP_GROUP}" /data; then
  echo "Warning: could not chown /data (common with id-mapped LXC mounts)."
  echo "Ensure the mount is writable by ${APP_USER} or configure shifted mount mapping."
fi

if [[ ! -d "${VENV_DIR}" ]]; then
  python3 -m venv "${VENV_DIR}"
fi

"${VENV_DIR}/bin/pip" install --upgrade pip
"${VENV_DIR}/bin/pip" install -r "${APP_DIR}/requirements.txt"

if [[ ! -f "${ENV_FILE}" ]]; then
  SESSION_SECRET="$(python3 - <<'PY'
import secrets
print(secrets.token_urlsafe(48))
PY
)"
  cat > "${ENV_FILE}" <<EOF_ENV
APP_NAME=HA Entity Vault
SESSION_SECRET=${SESSION_SECRET}
HEV_DATA_DIR=/data
HA_TOKEN=
EOF_ENV
  chmod 600 "${ENV_FILE}"
fi

install -m 0644 "${APP_DIR}/lxc/ha-entity-vault.service" "${SERVICE_FILE}"
systemctl daemon-reload
systemctl enable --now ha-entity-vault.service

systemctl --no-pager --full status ha-entity-vault.service || true
