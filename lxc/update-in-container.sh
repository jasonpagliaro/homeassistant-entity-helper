#!/usr/bin/env bash
set -euo pipefail

APP_DIR="/srv/ha-entity-vault"
VENV_DIR="/opt/ha-entity-vault/.venv"
SERVICE_NAME="ha-entity-vault.service"
APP_ENV_FILE="/etc/default/ha-entity-vault"

HEV_UPDATE_HEALTH_TIMEOUT_SEC="${HEV_UPDATE_HEALTH_TIMEOUT_SEC:-5}"
HEV_UPDATE_BACKUP_RETENTION="${HEV_UPDATE_BACKUP_RETENTION:-14}"

log() {
  local level="$1"
  shift
  printf "%s [%s] %s\n" "$(date -u +"%Y-%m-%dT%H:%M:%SZ")" "${level}" "$*" >&2
}

die() {
  local code="$1"
  shift
  log "ERROR" "$*"
  exit "${code}"
}

require_root() {
  if [[ "${EUID}" -ne 0 ]]; then
    die 1 "Run as root inside the LXC container."
  fi
}

load_app_env() {
  if [[ -f "${APP_ENV_FILE}" ]]; then
    # shellcheck disable=SC1090
    source "${APP_ENV_FILE}"
  fi
}

is_positive_integer() {
  local value="$1"
  [[ "${value}" =~ ^[0-9]+$ ]] && (( value > 0 ))
}

normalize_positive_integer() {
  local value="$1"
  local fallback="$2"
  if is_positive_integer "${value}"; then
    printf "%s\n" "${value}"
  else
    printf "%s\n" "${fallback}"
  fi
}

resolve_db_path() {
  load_app_env

  if [[ -n "${DATABASE_URL:-}" ]]; then
    case "${DATABASE_URL}" in
      sqlite:///*)
        printf "%s\n" "${DATABASE_URL#sqlite:///}"
        return
        ;;
      *)
        die 2 "Unsupported DATABASE_URL for backup/restore: ${DATABASE_URL}"
        ;;
    esac
  fi

  if [[ -n "${HEV_DB_PATH:-}" ]]; then
    printf "%s\n" "${HEV_DB_PATH}"
    return
  fi

  local data_dir="${HEV_DATA_DIR:-/data}"
  printf "%s\n" "${data_dir}/ha_entity_vault.db"
}

cmd_install() {
  if [[ ! -f "${APP_DIR}/requirements.txt" ]]; then
    die 3 "Missing ${APP_DIR}/requirements.txt."
  fi

  if [[ ! -d "${VENV_DIR}" ]]; then
    log "INFO" "Virtualenv missing at ${VENV_DIR}; creating it."
    python3 -m venv "${VENV_DIR}"
  fi

  "${VENV_DIR}/bin/pip" install --upgrade pip
  "${VENV_DIR}/bin/pip" install -r "${APP_DIR}/requirements.txt"
}

cmd_smoke() {
  local timeout
  timeout="$(normalize_positive_integer "${HEV_UPDATE_HEALTH_TIMEOUT_SEC}" 5)"

  systemctl is-active --quiet "${SERVICE_NAME}" || die 4 "${SERVICE_NAME} is not active."

  local health_payload
  health_payload="$(curl -fsS --max-time "${timeout}" http://127.0.0.1:8000/healthz)"
  if ! grep -Eq '"status"[[:space:]]*:[[:space:]]*"ok"' <<<"${health_payload}"; then
    die 5 "Unexpected /healthz payload: ${health_payload}"
  fi

  local entities_status
  entities_status="$(curl -sS --max-time "${timeout}" -o /dev/null -w "%{http_code}" http://127.0.0.1:8000/entities)"
  if [[ "${entities_status}" != "200" ]]; then
    die 6 "/entities returned ${entities_status}, expected 200."
  fi

  local root_status
  root_status="$(curl -sS --max-time "${timeout}" -o /dev/null -w "%{http_code}" http://127.0.0.1:8000/)"
  if [[ "${root_status}" != "303" ]]; then
    die 7 "/ returned ${root_status}, expected 303."
  fi

  sleep 10
  systemctl is-active --quiet "${SERVICE_NAME}" || die 8 "${SERVICE_NAME} became inactive after stabilization wait."
  log "INFO" "Smoke checks passed."
}

prune_backups() {
  local backup_dir="$1"
  local retention
  retention="$(normalize_positive_integer "${HEV_UPDATE_BACKUP_RETENTION}" 14)"

  local backups=()
  mapfile -t backups < <(ls -1t "${backup_dir}"/ha_entity_vault.db.*.bak 2>/dev/null || true)
  if (( ${#backups[@]} <= retention )); then
    return
  fi

  local idx
  for ((idx = retention; idx < ${#backups[@]}; idx++)); do
    rm -f "${backups[${idx}]}"
  done
}

cmd_backup_db() {
  local db_path
  db_path="$(resolve_db_path)"
  if [[ ! -f "${db_path}" ]]; then
    die 9 "Database file not found at ${db_path}."
  fi

  local db_dir backup_dir timestamp backup_path
  db_dir="$(dirname "${db_path}")"
  backup_dir="${db_dir}/backups"
  timestamp="$(date -u +"%Y%m%dT%H%M%SZ")"
  backup_path="${backup_dir}/ha_entity_vault.db.${timestamp}.bak"

  mkdir -p "${backup_dir}"
  cp "${db_path}" "${backup_path}"
  chmod 600 "${backup_path}" || true
  prune_backups "${backup_dir}"

  printf "%s\n" "${backup_path}"
  log "INFO" "Database backup created at ${backup_path}."
}

cmd_restore_db() {
  local backup_path="${1:-}"
  if [[ -z "${backup_path}" ]]; then
    die 10 "restore-db requires a backup path argument."
  fi
  if [[ ! -f "${backup_path}" ]]; then
    die 11 "Backup file does not exist: ${backup_path}"
  fi

  local db_path
  db_path="$(resolve_db_path)"
  mkdir -p "$(dirname "${db_path}")"
  cp "${backup_path}" "${db_path}"
  chmod 600 "${db_path}" || true
  log "INFO" "Database restored from ${backup_path} to ${db_path}."
}

usage() {
  cat <<EOF
Usage: $(basename "$0") <install|smoke|backup-db|restore-db>
EOF
}

main() {
  require_root

  local cmd="${1:-}"
  case "${cmd}" in
    install)
      cmd_install
      ;;
    smoke)
      cmd_smoke
      ;;
    backup-db)
      cmd_backup_db
      ;;
    restore-db)
      shift || true
      cmd_restore_db "${1:-}"
      ;;
    *)
      usage
      exit 64
      ;;
  esac
}

main "$@"
