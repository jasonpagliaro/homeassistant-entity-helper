#!/usr/bin/env bash
set -euo pipefail

PROD_REPO_PATH="${PROD_REPO_PATH:-/srv/ha-entity-vault-prod}"
PROD_CONTAINER="${PROD_CONTAINER:-ha-entity-vault}"
UPDATER_ENV_FILE="${UPDATER_ENV_FILE:-/etc/default/ha-entity-vault-updater}"
SYSTEMD_DIR="${SYSTEMD_DIR:-/etc/systemd/system}"
HEV_UPDATE_MODE_VALUE="${HEV_UPDATE_MODE_VALUE:-auto_apply}"
HEV_UPDATE_BRANCH_VALUE="${HEV_UPDATE_BRANCH_VALUE:-main}"
HEV_UPDATE_ALLOW_HIGH_RISK_AUTO_VALUE="${HEV_UPDATE_ALLOW_HIGH_RISK_AUTO_VALUE:-false}"
RUN_NOW="false"

log() {
  local level="$1"
  shift
  printf "%s [%s] %s\n" "$(date -u +"%Y-%m-%dT%H:%M:%SZ")" "${level}" "$*"
}

die() {
  local code="$1"
  shift
  log "ERROR" "$*"
  exit "${code}"
}

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Configure production updater to auto-apply main on schedule, while blocking high-risk auto updates.

Options:
  --prod-path PATH                  Production checkout path (default: ${PROD_REPO_PATH})
  --container NAME                  Production container name (default: ${PROD_CONTAINER})
  --updater-env-file PATH           Updater env path (default: ${UPDATER_ENV_FILE})
  --systemd-dir PATH                Systemd unit dir (default: ${SYSTEMD_DIR})
  --mode MODE                       HEV_UPDATE_MODE value (default: ${HEV_UPDATE_MODE_VALUE})
  --branch BRANCH                   HEV_UPDATE_BRANCH value (default: ${HEV_UPDATE_BRANCH_VALUE})
  --allow-high-risk-auto true|false HEV_UPDATE_ALLOW_HIGH_RISK_AUTO value (default: ${HEV_UPDATE_ALLOW_HIGH_RISK_AUTO_VALUE})
  --run-now                         Start updater service once after timer setup
  -h, --help                        Show this help message
EOF
}

require_command() {
  local command_name="$1"
  command -v "${command_name}" >/dev/null 2>&1 || die 2 "Missing required command: ${command_name}"
}

upsert_env() {
  local file="$1"
  local key="$2"
  local value="$3"

  if grep -qE "^${key}=" "${file}"; then
    sed -i "s|^${key}=.*|${key}=${value}|" "${file}"
  else
    printf "%s=%s\n" "${key}" "${value}" >> "${file}"
  fi
}

while (( "$#" )); do
  case "$1" in
    --prod-path)
      PROD_REPO_PATH="$2"
      shift 2
      ;;
    --container)
      PROD_CONTAINER="$2"
      shift 2
      ;;
    --updater-env-file)
      UPDATER_ENV_FILE="$2"
      shift 2
      ;;
    --systemd-dir)
      SYSTEMD_DIR="$2"
      shift 2
      ;;
    --mode)
      HEV_UPDATE_MODE_VALUE="$2"
      shift 2
      ;;
    --branch)
      HEV_UPDATE_BRANCH_VALUE="$2"
      shift 2
      ;;
    --allow-high-risk-auto)
      HEV_UPDATE_ALLOW_HIGH_RISK_AUTO_VALUE="$2"
      shift 2
      ;;
    --run-now)
      RUN_NOW="true"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      usage
      die 64 "Unknown argument: $1"
      ;;
  esac
done

if [[ "${EUID}" -ne 0 ]]; then
  die 1 "Run as root (sudo) to write ${UPDATER_ENV_FILE} and install systemd units."
fi

require_command install
require_command grep
require_command sed
require_command systemctl

if [[ ! -d "${PROD_REPO_PATH}/.git" ]]; then
  die 3 "Production checkout missing or not a git repo: ${PROD_REPO_PATH}"
fi

ENV_EXAMPLE="${PROD_REPO_PATH}/lxc/ha-entity-vault-update.env.example"
SERVICE_SRC="${PROD_REPO_PATH}/lxc/ha-entity-vault-update-check.service"
TIMER_SRC="${PROD_REPO_PATH}/lxc/ha-entity-vault-update-check.timer"

[[ -f "${ENV_EXAMPLE}" ]] || die 4 "Missing ${ENV_EXAMPLE}"
[[ -f "${SERVICE_SRC}" ]] || die 5 "Missing ${SERVICE_SRC}"
[[ -f "${TIMER_SRC}" ]] || die 6 "Missing ${TIMER_SRC}"

mkdir -p "$(dirname "${UPDATER_ENV_FILE}")"
if [[ ! -f "${UPDATER_ENV_FILE}" ]]; then
  install -m 0600 "${ENV_EXAMPLE}" "${UPDATER_ENV_FILE}"
fi

upsert_env "${UPDATER_ENV_FILE}" "HEV_UPDATE_MODE" "${HEV_UPDATE_MODE_VALUE}"
upsert_env "${UPDATER_ENV_FILE}" "HEV_UPDATE_CONTAINER" "${PROD_CONTAINER}"
upsert_env "${UPDATER_ENV_FILE}" "HEV_UPDATE_REPO_PATH" "${PROD_REPO_PATH}"
upsert_env "${UPDATER_ENV_FILE}" "HEV_UPDATE_BRANCH" "${HEV_UPDATE_BRANCH_VALUE}"
upsert_env "${UPDATER_ENV_FILE}" "HEV_UPDATE_ALLOW_HIGH_RISK_AUTO" "${HEV_UPDATE_ALLOW_HIGH_RISK_AUTO_VALUE}"
chmod 600 "${UPDATER_ENV_FILE}"

install -m 0644 "${SERVICE_SRC}" "${SYSTEMD_DIR}/ha-entity-vault-update-check.service"
install -m 0644 "${TIMER_SRC}" "${SYSTEMD_DIR}/ha-entity-vault-update-check.timer"

systemctl daemon-reload
systemctl enable --now ha-entity-vault-update-check.timer

if [[ "${RUN_NOW}" == "true" ]]; then
  systemctl start ha-entity-vault-update-check.service
fi

cat <<EOF

Production updater configured.
Env file: ${UPDATER_ENV_FILE}
Repo:     ${PROD_REPO_PATH}
Branch:   ${HEV_UPDATE_BRANCH_VALUE}
Mode:     ${HEV_UPDATE_MODE_VALUE}

Verify:
1. systemctl status ha-entity-vault-update-check.timer --no-pager
2. journalctl -u ha-entity-vault-update-check.service -n 100 --no-pager
3. HEV_UPDATE_ENV_FILE=${UPDATER_ENV_FILE} ${PROD_REPO_PATH}/lxc/update-host.sh status
EOF
