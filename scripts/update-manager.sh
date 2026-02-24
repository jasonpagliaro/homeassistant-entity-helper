#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_FILE="${ENV_FILE:-${REPO_ROOT}/.env}"

AUTO_UPDATE_ENABLED="${AUTO_UPDATE_ENABLED:-true}"
AUTO_UPDATE_SCHEDULE="${AUTO_UPDATE_SCHEDULE:-04:00}"
UPDATE_TIMEOUT_SECONDS="${UPDATE_TIMEOUT_SECONDS:-120}"
BACKUP_RETENTION_DAYS="${BACKUP_RETENTION_DAYS:-7}"
UPDATE_BRANCH="${UPDATE_BRANCH:-main}"

AUTO_UPDATE_LOG_FILE="${AUTO_UPDATE_LOG_FILE:-/var/log/ha-entity-vault-update.log}"
AUTO_UPDATE_BACKUP_DIR="${AUTO_UPDATE_BACKUP_DIR:-/var/backups/ha-entity-vault}"
AUTO_UPDATE_STATE_DIR="${AUTO_UPDATE_STATE_DIR:-/var/lib/ha-entity-vault-update}"
AUTO_UPDATE_LOCK_FILE="${AUTO_UPDATE_LOCK_FILE:-/var/lock/ha-entity-vault-update.lock}"
AUTO_UPDATE_MIN_FREE_MB="${AUTO_UPDATE_MIN_FREE_MB:-1024}"
UPDATE_POLL_SECONDS="${UPDATE_POLL_SECONDS:-3}"
UPDATE_EXPECTED_ORIGIN="${UPDATE_EXPECTED_ORIGIN:-https://github.com/jasonpagliaro/homeassistant-entity-helper.git}"
COMPOSE_SERVICE="${COMPOSE_SERVICE:-app}"
SQLITE_VOLUME="${SQLITE_VOLUME:-hev_data}"
SQLITE_DB_PATH="${SQLITE_DB_PATH:-/data/ha_entity_vault.db}"
POST_DEPLOY_MONITOR_SECONDS="${POST_DEPLOY_MONITOR_SECONDS:-600}"
POST_DEPLOY_CRASH_THRESHOLD="${POST_DEPLOY_CRASH_THRESHOLD:-3}"

METADATA_FILE=""

CURRENT_SHA=""
LKG_SHA=""
UPDATES_PAUSED="false"
PAUSE_REASON=""
LAST_SCHEDULED_RUN_DATE=""
LAST_UPDATE_ATTEMPT_AT=""
LAST_UPDATE_RESULT="never"
LAST_BACKUP_PATH=""
LAST_DURATION_SECONDS="0"
LAST_TARGET_SHA=""
LAST_ROLLBACK_REASON=""

ATTEMPT_STARTED_AT=""
ATTEMPT_STARTED_EPOCH="0"
ATTEMPT_CURRENT_VERSION=""
ATTEMPT_TARGET_VERSION=""
ATTEMPT_BACKUP_PATH=""
ATTEMPT_OUTCOME="never"
ATTEMPT_ROLLBACK="false"
ATTEMPT_REASON=""

LOCK_FD=9
LOCK_ACQUIRED="false"

usage() {
  cat <<'EOF_USAGE'
Usage:
  scripts/update-manager.sh run --scheduled
  scripts/update-manager.sh run --force
  scripts/update-manager.sh rollback --reason "manual rollback"
  scripts/update-manager.sh resume-updates
  scripts/update-manager.sh status
EOF_USAGE
}

trim() {
  local value="$1"
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  printf '%s' "${value}"
}

load_env_file() {
  local file_path="$1"
  local line key value first_char last_char

  if [[ ! -f "${file_path}" ]]; then
    return
  fi

  while IFS= read -r line || [[ -n "${line}" ]]; do
    line="${line%%$'\r'}"
    [[ -z "$(trim "${line}")" ]] && continue
    [[ "${line}" =~ ^[[:space:]]*# ]] && continue
    [[ "${line}" != *=* ]] && continue

    key="${line%%=*}"
    value="${line#*=}"
    key="$(trim "${key}")"
    value="$(trim "${value}")"
    key="${key#export }"

    if [[ ! "${key}" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]]; then
      continue
    fi

    if (( ${#value} >= 2 )); then
      first_char="${value:0:1}"
      last_char="${value: -1}"
      if [[ ( "${first_char}" == "\"" && "${last_char}" == "\"" ) || ( "${first_char}" == "'" && "${last_char}" == "'" ) ]]; then
        value="${value:1:$(( ${#value} - 2 ))}"
      fi
    fi

    printf -v "${key}" '%s' "${value}"
    export "${key}"
  done < "${file_path}"
}

is_truthy() {
  local normalized
  normalized="$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')"
  case "${normalized}" in
    1|true|yes|on)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

require_cmd() {
  local command_name="$1"
  if ! command -v "${command_name}" >/dev/null 2>&1; then
    printf 'Missing required command: %s\n' "${command_name}" >&2
    exit 2
  fi
}

escape_json() {
  local value="$1"
  value="${value//\\/\\\\}"
  value="${value//\"/\\\"}"
  value="${value//$'\n'/\\n}"
  value="${value//$'\r'/\\r}"
  value="${value//$'\t'/\\t}"
  printf '%s' "${value}"
}

now_utc_iso() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

log_json() {
  local level="$1"
  local event="$2"
  shift 2

  local timestamp payload key value
  timestamp="$(now_utc_iso)"
  payload="{\"timestamp\":\"$(escape_json "${timestamp}")\",\"level\":\"$(escape_json "${level}")\",\"event\":\"$(escape_json "${event}")\""

  while (( $# >= 2 )); do
    key="$1"
    value="$2"
    shift 2
    payload+=",\"$(escape_json "${key}")\":\"$(escape_json "${value}")\""
  done

  payload+="}"
  printf '%s\n' "${payload}" | tee -a "${AUTO_UPDATE_LOG_FILE}" >/dev/null
}

apply_defaults() {
  AUTO_UPDATE_ENABLED="${AUTO_UPDATE_ENABLED:-true}"
  AUTO_UPDATE_SCHEDULE="${AUTO_UPDATE_SCHEDULE:-04:00}"
  UPDATE_TIMEOUT_SECONDS="${UPDATE_TIMEOUT_SECONDS:-120}"
  BACKUP_RETENTION_DAYS="${BACKUP_RETENTION_DAYS:-7}"
  UPDATE_BRANCH="${UPDATE_BRANCH:-main}"

  AUTO_UPDATE_LOG_FILE="${AUTO_UPDATE_LOG_FILE:-/var/log/ha-entity-vault-update.log}"
  AUTO_UPDATE_BACKUP_DIR="${AUTO_UPDATE_BACKUP_DIR:-/var/backups/ha-entity-vault}"
  AUTO_UPDATE_STATE_DIR="${AUTO_UPDATE_STATE_DIR:-/var/lib/ha-entity-vault-update}"
  AUTO_UPDATE_LOCK_FILE="${AUTO_UPDATE_LOCK_FILE:-/var/lock/ha-entity-vault-update.lock}"
  AUTO_UPDATE_MIN_FREE_MB="${AUTO_UPDATE_MIN_FREE_MB:-1024}"
  UPDATE_POLL_SECONDS="${UPDATE_POLL_SECONDS:-3}"
  UPDATE_EXPECTED_ORIGIN="${UPDATE_EXPECTED_ORIGIN:-https://github.com/jasonpagliaro/homeassistant-entity-helper.git}"
  COMPOSE_SERVICE="${COMPOSE_SERVICE:-app}"
  SQLITE_VOLUME="${SQLITE_VOLUME:-hev_data}"
  SQLITE_DB_PATH="${SQLITE_DB_PATH:-/data/ha_entity_vault.db}"
  POST_DEPLOY_MONITOR_SECONDS="${POST_DEPLOY_MONITOR_SECONDS:-600}"
  POST_DEPLOY_CRASH_THRESHOLD="${POST_DEPLOY_CRASH_THRESHOLD:-3}"
}

ensure_runtime_paths() {
  mkdir -p "$(dirname "${AUTO_UPDATE_LOG_FILE}")"
  mkdir -p "${AUTO_UPDATE_BACKUP_DIR}"
  mkdir -p "${AUTO_UPDATE_STATE_DIR}"
  mkdir -p "$(dirname "${AUTO_UPDATE_LOCK_FILE}")"
  touch "${AUTO_UPDATE_LOG_FILE}"
}

load_metadata() {
  METADATA_FILE="${AUTO_UPDATE_STATE_DIR}/metadata.env"
  if [[ -f "${METADATA_FILE}" ]]; then
    # shellcheck disable=SC1090
    source "${METADATA_FILE}"
  fi

  CURRENT_SHA="${CURRENT_SHA:-}"
  LKG_SHA="${LKG_SHA:-}"
  UPDATES_PAUSED="${UPDATES_PAUSED:-false}"
  PAUSE_REASON="${PAUSE_REASON:-}"
  LAST_SCHEDULED_RUN_DATE="${LAST_SCHEDULED_RUN_DATE:-}"
  LAST_UPDATE_ATTEMPT_AT="${LAST_UPDATE_ATTEMPT_AT:-}"
  LAST_UPDATE_RESULT="${LAST_UPDATE_RESULT:-never}"
  LAST_BACKUP_PATH="${LAST_BACKUP_PATH:-}"
  LAST_DURATION_SECONDS="${LAST_DURATION_SECONDS:-0}"
  LAST_TARGET_SHA="${LAST_TARGET_SHA:-}"
  LAST_ROLLBACK_REASON="${LAST_ROLLBACK_REASON:-}"
}

write_metadata() {
  local tmp_file
  tmp_file="$(mktemp "${AUTO_UPDATE_STATE_DIR}/metadata.XXXXXX")"
  {
    printf 'CURRENT_SHA=%q\n' "${CURRENT_SHA}"
    printf 'LKG_SHA=%q\n' "${LKG_SHA}"
    printf 'UPDATES_PAUSED=%q\n' "${UPDATES_PAUSED}"
    printf 'PAUSE_REASON=%q\n' "${PAUSE_REASON}"
    printf 'LAST_SCHEDULED_RUN_DATE=%q\n' "${LAST_SCHEDULED_RUN_DATE}"
    printf 'LAST_UPDATE_ATTEMPT_AT=%q\n' "${LAST_UPDATE_ATTEMPT_AT}"
    printf 'LAST_UPDATE_RESULT=%q\n' "${LAST_UPDATE_RESULT}"
    printf 'LAST_BACKUP_PATH=%q\n' "${LAST_BACKUP_PATH}"
    printf 'LAST_DURATION_SECONDS=%q\n' "${LAST_DURATION_SECONDS}"
    printf 'LAST_TARGET_SHA=%q\n' "${LAST_TARGET_SHA}"
    printf 'LAST_ROLLBACK_REASON=%q\n' "${LAST_ROLLBACK_REASON}"
  } > "${tmp_file}"
  mv "${tmp_file}" "${METADATA_FILE}"
}

normalize_remote_url() {
  local remote="$1"
  remote="${remote%.git}"
  if [[ "${remote}" =~ ^git@github\.com:(.+)$ ]]; then
    printf 'https://github.com/%s\n' "${BASH_REMATCH[1]}"
    return
  fi
  if [[ "${remote}" =~ ^https://github\.com/(.+)$ ]]; then
    printf 'https://github.com/%s\n' "${BASH_REMATCH[1]}"
    return
  fi
  printf '%s\n' "${remote}"
}

acquire_lock() {
  exec {LOCK_FD}>"${AUTO_UPDATE_LOCK_FILE}"
  if ! flock -n "${LOCK_FD}"; then
    return 1
  fi
  LOCK_ACQUIRED="true"
  return 0
}

release_lock() {
  if [[ "${LOCK_ACQUIRED}" == "true" ]]; then
    flock -u "${LOCK_FD}" || true
    LOCK_ACQUIRED="false"
  fi
}

validate_numeric_settings() {
  local value_name value
  for value_name in UPDATE_TIMEOUT_SECONDS BACKUP_RETENTION_DAYS AUTO_UPDATE_MIN_FREE_MB UPDATE_POLL_SECONDS POST_DEPLOY_MONITOR_SECONDS POST_DEPLOY_CRASH_THRESHOLD; do
    value="${!value_name}"
    if [[ ! "${value}" =~ ^[0-9]+$ ]]; then
      log_json "ERROR" "invalid_configuration" "field" "${value_name}" "value" "${value}"
      exit 2
    fi
  done
}

require_run_dependencies() {
  require_cmd git
  require_cmd docker
  require_cmd flock
  require_cmd find
  require_cmd awk
}

require_rollback_dependencies() {
  require_cmd docker
  require_cmd flock
}

require_resume_dependencies() {
  require_cmd flock
}

compose_container_id() {
  docker compose ps -q "${COMPOSE_SERVICE}" | head -n 1
}

container_health() {
  local container_id="$1"
  docker inspect --format '{{if .State.Health}}{{.State.Health.Status}}{{else}}{{.State.Status}}{{end}}' "${container_id}"
}

container_user() {
  local container_id="$1"
  docker inspect --format '{{.Config.User}}' "${container_id}"
}

container_restart_count() {
  local container_id="$1"
  docker inspect --format '{{.RestartCount}}' "${container_id}"
}

wait_for_service_healthy() {
  local timeout_seconds="$1"
  local deadline now container_id health_status

  deadline=$(( $(date +%s) + timeout_seconds ))
  while true; do
    now="$(date +%s)"
    if (( now >= deadline )); then
      return 1
    fi

    container_id="$(compose_container_id)"
    if [[ -z "${container_id}" ]]; then
      sleep "${UPDATE_POLL_SECONDS}"
      continue
    fi

    health_status="$(container_health "${container_id}" 2>/dev/null || true)"
    if [[ "${health_status}" == "healthy" ]]; then
      return 0
    fi
    if [[ "${health_status}" == "unhealthy" || "${health_status}" == "exited" || "${health_status}" == "dead" ]]; then
      return 1
    fi
    sleep "${UPDATE_POLL_SECONDS}"
  done
}

get_alembic_version() {
  docker run --rm \
    -v "${SQLITE_VOLUME}:/data:ro" \
    python:3.12-slim \
    python - "${SQLITE_DB_PATH}" <<'PY'
import os
import sqlite3
import sys

db_path = sys.argv[1]
if not os.path.exists(db_path):
    print("")
    raise SystemExit(0)

conn = sqlite3.connect(db_path)
try:
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='alembic_version'")
    if cur.fetchone() is None:
        print("")
    else:
        cur.execute("SELECT version_num FROM alembic_version LIMIT 1")
        row = cur.fetchone()
        print(row[0] if row and row[0] else "")
finally:
    conn.close()
PY
}

backup_sqlite() {
  local backup_path="$1"
  local backup_name
  backup_name="$(basename "${backup_path}")"

  docker run --rm \
    -v "${SQLITE_VOLUME}:/data:ro" \
    -v "${AUTO_UPDATE_BACKUP_DIR}:/backup:rw" \
    python:3.12-slim \
    python - "${SQLITE_DB_PATH}" "/backup/${backup_name}" <<'PY'
import os
import sqlite3
import sys

source_path = sys.argv[1]
dest_path = sys.argv[2]
if not os.path.exists(source_path):
    raise SystemExit("sqlite_source_missing")

with sqlite3.connect(source_path, timeout=30) as source_conn:
    with sqlite3.connect(dest_path, timeout=30) as dest_conn:
        source_conn.backup(dest_conn)
PY
}

restore_sqlite_backup() {
  local backup_path="$1"
  local backup_name
  backup_name="$(basename "${backup_path}")"

  docker run --rm \
    -v "${SQLITE_VOLUME}:/data:rw" \
    -v "${AUTO_UPDATE_BACKUP_DIR}:/backup:ro" \
    python:3.12-slim \
    python - "${SQLITE_DB_PATH}" "/backup/${backup_name}" <<'PY'
import os
import shutil
import sys

db_path = sys.argv[1]
backup_path = sys.argv[2]
if not os.path.exists(backup_path):
    raise SystemExit("sqlite_backup_missing")

os.makedirs(os.path.dirname(db_path), exist_ok=True)
shutil.copy2(backup_path, db_path)
PY
}

update_db_update_status() {
  local attempt_at="$1"
  local update_result="$2"

  docker run --rm \
    -v "${SQLITE_VOLUME}:/data:rw" \
    python:3.12-slim \
    python - "${SQLITE_DB_PATH}" "${attempt_at}" "${update_result}" <<'PY'
import datetime
import os
import sqlite3
import sys

db_path, attempt_at, update_result = sys.argv[1:4]
if not os.path.exists(db_path):
    raise SystemExit(0)

conn = sqlite3.connect(db_path)
try:
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='app_config'")
    if cur.fetchone() is None:
        raise SystemExit(0)

    cur.execute("SELECT id FROM app_config ORDER BY id ASC LIMIT 1")
    row = cur.fetchone()
    now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
    if row is None:
        cur.execute(
            """
            INSERT INTO app_config (
              updates_enabled,
              update_repo_owner,
              update_repo_name,
              update_repo_branch,
              update_check_interval_minutes,
              last_checked_at,
              last_check_state,
              last_check_error,
              installed_commit_sha,
              latest_commit_sha,
              latest_commit_url,
              latest_commit_published_at,
              dismissed_commit_sha,
              dismissed_at,
              last_update_attempt_at,
              last_update_result,
              created_at,
              updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                1,
                "jasonpagliaro",
                "homeassistant-entity-helper",
                "main",
                720,
                None,
                "never",
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                attempt_at,
                update_result,
                now_iso,
                now_iso,
            ),
        )
        app_config_id = cur.lastrowid
    else:
        app_config_id = int(row[0])

    cur.execute(
        """
        UPDATE app_config
        SET last_update_attempt_at = ?, last_update_result = ?, updated_at = ?
        WHERE id = ?
        """,
        (attempt_at, update_result, now_iso, app_config_id),
    )
    conn.commit()
finally:
    conn.close()
PY
}

record_update_result() {
  local update_result="$1"
  local attempt_at="$2"

  LAST_UPDATE_ATTEMPT_AT="${attempt_at}"
  LAST_UPDATE_RESULT="${update_result}"
  write_metadata

  if ! update_db_update_status "${attempt_at}" "${update_result}" >/dev/null 2>&1; then
    log_json "WARNING" "db_status_update_failed" "result" "${update_result}"
  fi
}

run_prechecks() {
  local available_mb origin_url expected_origin current_branch container_id health_status user_value

  if ! docker info >/dev/null 2>&1; then
    log_json "ERROR" "precheck_failed" "check" "docker_daemon"
    return 1
  fi

  if ! origin_url="$(git remote get-url origin 2>/dev/null)"; then
    log_json "ERROR" "precheck_failed" "check" "git_origin_read"
    return 1
  fi
  expected_origin="$(normalize_remote_url "${UPDATE_EXPECTED_ORIGIN}")"
  if [[ "$(normalize_remote_url "${origin_url}")" != "${expected_origin}" ]]; then
    log_json "ERROR" "precheck_failed" "check" "git_origin" "origin_url" "${origin_url}" "expected" "${expected_origin}"
    return 1
  fi

  if ! git ls-remote --heads origin "${UPDATE_BRANCH}" >/dev/null 2>&1; then
    log_json "ERROR" "precheck_failed" "check" "git_remote_branch" "branch" "${UPDATE_BRANCH}"
    return 1
  fi

  if ! git diff --quiet || ! git diff --cached --quiet; then
    log_json "ERROR" "precheck_failed" "check" "git_clean_tree"
    return 1
  fi

  if ! current_branch="$(git rev-parse --abbrev-ref HEAD 2>/dev/null)"; then
    log_json "ERROR" "precheck_failed" "check" "git_branch_resolve"
    return 1
  fi
  if [[ "${current_branch}" != "${UPDATE_BRANCH}" ]]; then
    log_json "ERROR" "precheck_failed" "check" "git_branch" "current_branch" "${current_branch}" "expected_branch" "${UPDATE_BRANCH}"
    return 1
  fi

  available_mb="$(df -Pm "${AUTO_UPDATE_BACKUP_DIR}" | awk 'NR==2 {print $4}')"
  if [[ -z "${available_mb}" || ! "${available_mb}" =~ ^[0-9]+$ ]]; then
    log_json "ERROR" "precheck_failed" "check" "disk_space_parse"
    return 1
  fi
  if (( available_mb < AUTO_UPDATE_MIN_FREE_MB )); then
    log_json "ERROR" "precheck_failed" "check" "disk_space" "available_mb" "${available_mb}" "required_mb" "${AUTO_UPDATE_MIN_FREE_MB}"
    return 1
  fi

  container_id="$(compose_container_id)"
  if [[ -z "${container_id}" ]]; then
    log_json "ERROR" "precheck_failed" "check" "compose_container" "service" "${COMPOSE_SERVICE}"
    return 1
  fi

  health_status="$(container_health "${container_id}")"
  if [[ "${health_status}" != "healthy" ]]; then
    log_json "ERROR" "precheck_failed" "check" "container_health" "status" "${health_status}"
    return 1
  fi

  user_value="$(container_user "${container_id}")"
  case "${user_value}" in
    ""|0|0:0|root|root:root)
      log_json "ERROR" "precheck_failed" "check" "container_user" "user" "${user_value}"
      return 1
      ;;
    *)
      ;;
  esac

  if [[ "${SQLITE_DB_PATH}" != /data/* ]]; then
    log_json "ERROR" "precheck_failed" "check" "sqlite_db_path" "path" "${SQLITE_DB_PATH}"
    return 1
  fi

  if ! docker volume inspect "${SQLITE_VOLUME}" >/dev/null 2>&1; then
    log_json "ERROR" "precheck_failed" "check" "sqlite_volume" "volume" "${SQLITE_VOLUME}"
    return 1
  fi
  if ! docker run --rm -v "${SQLITE_VOLUME}:/data:ro" busybox sh -c 'test -r "$1" && test -s "$1"' _ "${SQLITE_DB_PATH}" >/dev/null 2>&1; then
    log_json "ERROR" "precheck_failed" "check" "sqlite_volume_file" "volume" "${SQLITE_VOLUME}" "path" "${SQLITE_DB_PATH}"
    return 1
  fi

  return 0
}

schedule_is_due() {
  local schedule_hour schedule_minute now_hour now_minute schedule_total now_total today

  if [[ ! "${AUTO_UPDATE_SCHEDULE}" =~ ^([01][0-9]|2[0-3]):([0-5][0-9])$ ]]; then
    log_json "ERROR" "invalid_configuration" "field" "AUTO_UPDATE_SCHEDULE" "value" "${AUTO_UPDATE_SCHEDULE}"
    return 2
  fi

  schedule_hour="${BASH_REMATCH[1]}"
  schedule_minute="${BASH_REMATCH[2]}"
  now_hour="$(date +%H)"
  now_minute="$(date +%M)"
  schedule_total=$(( 10#${schedule_hour} * 60 + 10#${schedule_minute} ))
  now_total=$(( 10#${now_hour} * 60 + 10#${now_minute} ))
  today="$(date +%F)"

  if [[ "${LAST_SCHEDULED_RUN_DATE}" == "${today}" ]]; then
    return 1
  fi
  if (( now_total < schedule_total )); then
    return 1
  fi
  return 0
}

prune_old_backups() {
  find "${AUTO_UPDATE_BACKUP_DIR}" -maxdepth 1 -type f -name 'ha_entity_vault_*.db' -mtime +"${BACKUP_RETENTION_DAYS}" -delete
}

rollback_to_last_known_good() {
  local reason="$1"
  local pre_alembic_version="$2"
  local backup_path="$3"
  local post_alembic_version backup_restore_required backup_restore_ok

  LAST_ROLLBACK_REASON="${reason}"
  write_metadata
  ATTEMPT_ROLLBACK="true"

  if ! docker image inspect "ha-entity-vault:last-known-good" >/dev/null 2>&1; then
    log_json "ERROR" "rollback_failed" "reason" "${reason}" "detail" "last_known_good_image_missing"
    return 1
  fi

  docker compose stop "${COMPOSE_SERVICE}" >/dev/null 2>&1 || true

  backup_restore_required="false"
  backup_restore_ok="true"
  post_alembic_version="$(get_alembic_version 2>/dev/null || true)"
  if [[ -n "${pre_alembic_version}" && "${post_alembic_version}" != "${pre_alembic_version}" ]]; then
    backup_restore_required="true"
  fi

  if [[ "${backup_restore_required}" == "true" && -n "${backup_path}" && -f "${backup_path}" ]]; then
    if ! restore_sqlite_backup "${backup_path}"; then
      backup_restore_ok="false"
      log_json "ERROR" "rollback_failed" "reason" "${reason}" "detail" "sqlite_restore_failed" "backup_path" "${backup_path}"
    else
      log_json "WARNING" "sqlite_restored_for_rollback" "reason" "${reason}" "backup_path" "${backup_path}"
    fi
  fi

  if [[ "${backup_restore_ok}" != "true" ]]; then
    return 1
  fi

  if ! HEV_IMAGE_TAG="ha-entity-vault:last-known-good" \
    HEV_BUILD_COMMIT_SHA="${LKG_SHA}" \
    docker compose up -d --no-deps --no-build "${COMPOSE_SERVICE}"; then
    log_json "ERROR" "rollback_failed" "reason" "${reason}" "detail" "compose_up_failed"
    return 1
  fi

  if ! wait_for_service_healthy "${UPDATE_TIMEOUT_SECONDS}"; then
    log_json "ERROR" "rollback_failed" "reason" "${reason}" "detail" "healthcheck_failed"
    return 1
  fi

  if [[ -n "${LKG_SHA}" ]]; then
    CURRENT_SHA="${LKG_SHA}"
    write_metadata
  fi

  log_json "WARNING" "rollback_succeeded" "reason" "${reason}" "lkg_sha" "${LKG_SHA}"
  return 0
}

finish_attempt() {
  local return_code="$1"
  local duration_seconds

  duration_seconds=$(( $(date +%s) - ATTEMPT_STARTED_EPOCH ))
  LAST_DURATION_SECONDS="${duration_seconds}"
  LAST_TARGET_SHA="${ATTEMPT_TARGET_VERSION}"
  LAST_BACKUP_PATH="${ATTEMPT_BACKUP_PATH}"
  write_metadata
  record_update_result "${ATTEMPT_OUTCOME}" "${ATTEMPT_STARTED_AT}"

  log_json "INFO" "update_attempt_completed" \
    "start_time" "${ATTEMPT_STARTED_AT}" \
    "current_version" "${ATTEMPT_CURRENT_VERSION}" \
    "target_version" "${ATTEMPT_TARGET_VERSION}" \
    "backup_path" "${ATTEMPT_BACKUP_PATH}" \
    "outcome" "${ATTEMPT_OUTCOME}" \
    "duration_seconds" "${duration_seconds}" \
    "rollback" "${ATTEMPT_ROLLBACK}" \
    "reason" "${ATTEMPT_REASON}"

  return "${return_code}"
}

set_updates_paused() {
  local reason="$1"
  UPDATES_PAUSED="true"
  PAUSE_REASON="${reason}"
  write_metadata
}

run_update() {
  local run_mode="$1"
  local now_date repo_head_sha target_sha backup_stamp backup_file backup_path candidate_tag running_container_id running_image_id
  local pre_alembic_version post_monitor_deadline baseline_restart_count current_restart_count restart_delta
  local schedule_check_status

  ATTEMPT_STARTED_AT="$(now_utc_iso)"
  ATTEMPT_STARTED_EPOCH="$(date +%s)"
  ATTEMPT_TARGET_VERSION=""
  ATTEMPT_BACKUP_PATH=""
  ATTEMPT_ROLLBACK="false"
  ATTEMPT_OUTCOME="error"
  ATTEMPT_REASON=""

  repo_head_sha="$(git rev-parse HEAD)"
  if [[ -z "${CURRENT_SHA}" ]]; then
    CURRENT_SHA="${repo_head_sha}"
    write_metadata
  fi
  ATTEMPT_CURRENT_VERSION="${CURRENT_SHA}"

  if ! is_truthy "${AUTO_UPDATE_ENABLED}"; then
    ATTEMPT_OUTCOME="skipped_disabled"
    ATTEMPT_REASON="AUTO_UPDATE_ENABLED=false"
    finish_attempt 0
    return $?
  fi

  if is_truthy "${UPDATES_PAUSED}"; then
    ATTEMPT_OUTCOME="skipped_paused"
    ATTEMPT_REASON="${PAUSE_REASON:-updates_paused}"
    finish_attempt 0
    return $?
  fi

  if [[ "${run_mode}" == "scheduled" ]]; then
    if schedule_is_due; then
      :
    else
      schedule_check_status=$?
      if (( schedule_check_status == 2 )); then
        ATTEMPT_OUTCOME="invalid_schedule"
        ATTEMPT_REASON="AUTO_UPDATE_SCHEDULE_invalid"
        finish_attempt 1
        return $?
      fi
      log_json "INFO" "scheduled_run_not_due" "schedule" "${AUTO_UPDATE_SCHEDULE}" "last_scheduled_run_date" "${LAST_SCHEDULED_RUN_DATE}"
      return 0
    fi
    now_date="$(date +%F)"
    LAST_SCHEDULED_RUN_DATE="${now_date}"
    write_metadata
  fi

  log_json "INFO" "update_attempt_started" \
    "start_time" "${ATTEMPT_STARTED_AT}" \
    "current_version" "${ATTEMPT_CURRENT_VERSION}" \
    "branch" "${UPDATE_BRANCH}" \
    "mode" "${run_mode}"

  if ! run_prechecks; then
    ATTEMPT_OUTCOME="precheck_failed"
    ATTEMPT_REASON="one_or_more_prechecks_failed"
    finish_attempt 1
    return $?
  fi

  git fetch --prune origin
  repo_head_sha="$(git rev-parse HEAD)"
  target_sha="$(git rev-parse "origin/${UPDATE_BRANCH}")"
  ATTEMPT_TARGET_VERSION="${target_sha}"

  if [[ "${target_sha}" == "${CURRENT_SHA}" ]]; then
    ATTEMPT_OUTCOME="no_update"
    ATTEMPT_REASON="target_sha_matches_current_sha"
    finish_attempt 0
    return $?
  fi

  if ! git merge-base --is-ancestor "${repo_head_sha}" "${target_sha}"; then
    ATTEMPT_OUTCOME="non_fast_forward"
    ATTEMPT_REASON="local_branch_cannot_fast_forward_to_target"
    finish_attempt 1
    return $?
  fi

  if ! git pull --ff-only origin "${UPDATE_BRANCH}"; then
    ATTEMPT_OUTCOME="git_pull_failed"
    ATTEMPT_REASON="git_pull_ff_only_failed"
    finish_attempt 1
    return $?
  fi

  backup_stamp="$(date -u +%Y%m%dT%H%M%SZ)"
  backup_file="ha_entity_vault_${backup_stamp}_${CURRENT_SHA:0:12}.db"
  backup_path="${AUTO_UPDATE_BACKUP_DIR}/${backup_file}"
  ATTEMPT_BACKUP_PATH="${backup_path}"

  pre_alembic_version="$(get_alembic_version 2>/dev/null || true)"
  if ! backup_sqlite "${backup_path}"; then
    ATTEMPT_OUTCOME="backup_failed"
    ATTEMPT_REASON="sqlite_backup_failed"
    finish_attempt 1
    return $?
  fi
  if [[ ! -s "${backup_path}" ]]; then
    ATTEMPT_OUTCOME="backup_invalid"
    ATTEMPT_REASON="backup_file_missing_or_empty"
    finish_attempt 1
    return $?
  fi

  prune_old_backups

  candidate_tag="ha-entity-vault:${target_sha:0:12}"
  if ! docker build --build-arg "HEV_BUILD_COMMIT_SHA=${target_sha}" -t "${candidate_tag}" .; then
    ATTEMPT_OUTCOME="build_failed"
    ATTEMPT_REASON="docker_build_failed"
    finish_attempt 1
    return $?
  fi

  running_container_id="$(compose_container_id)"
  if [[ -z "${running_container_id}" ]]; then
    ATTEMPT_OUTCOME="deploy_failed"
    ATTEMPT_REASON="running_container_missing_before_deploy"
    finish_attempt 1
    return $?
  fi
  running_image_id="$(docker inspect --format '{{.Image}}' "${running_container_id}")"
  docker image tag "${running_image_id}" "ha-entity-vault:last-known-good"
  LKG_SHA="${CURRENT_SHA}"
  write_metadata

  if ! HEV_IMAGE_TAG="${candidate_tag}" \
    HEV_BUILD_COMMIT_SHA="${target_sha}" \
    docker compose up -d --no-deps --no-build "${COMPOSE_SERVICE}"; then
    ATTEMPT_REASON="compose_up_candidate_failed"
    ATTEMPT_OUTCOME="deploy_failed"
    if rollback_to_last_known_good "candidate_deploy_failed" "${pre_alembic_version}" "${backup_path}"; then
      ATTEMPT_OUTCOME="rolled_back_after_failed_deploy"
      finish_attempt 1
      return $?
    fi
    set_updates_paused "rollback_failed_after_candidate_deploy"
    ATTEMPT_OUTCOME="catastrophic_rollback_failed"
    ATTEMPT_REASON="rollback_failed_after_candidate_deploy"
    finish_attempt 1
    return $?
  fi

  if ! wait_for_service_healthy "${UPDATE_TIMEOUT_SECONDS}"; then
    ATTEMPT_REASON="candidate_healthcheck_timeout_or_failure"
    ATTEMPT_OUTCOME="deploy_health_failed"
    if rollback_to_last_known_good "candidate_health_failed" "${pre_alembic_version}" "${backup_path}"; then
      ATTEMPT_OUTCOME="rolled_back_after_failed_deploy"
      finish_attempt 1
      return $?
    fi
    set_updates_paused "rollback_failed_after_health_failure"
    ATTEMPT_OUTCOME="catastrophic_rollback_failed"
    ATTEMPT_REASON="rollback_failed_after_health_failure"
    finish_attempt 1
    return $?
  fi

  docker image tag "${candidate_tag}" "ha-entity-vault:current"
  CURRENT_SHA="${target_sha}"
  write_metadata

  running_container_id="$(compose_container_id)"
  baseline_restart_count="$(container_restart_count "${running_container_id}")"
  post_monitor_deadline=$(( $(date +%s) + POST_DEPLOY_MONITOR_SECONDS ))

  while (( $(date +%s) < post_monitor_deadline )); do
    running_container_id="$(compose_container_id)"
    if [[ -z "${running_container_id}" ]]; then
      ATTEMPT_REASON="container_missing_during_post_deploy_monitor"
      ATTEMPT_OUTCOME="post_deploy_monitor_failed"
      break
    fi

    if [[ "$(container_health "${running_container_id}" 2>/dev/null || true)" == "unhealthy" ]]; then
      ATTEMPT_REASON="container_unhealthy_during_post_deploy_monitor"
      ATTEMPT_OUTCOME="post_deploy_monitor_failed"
      break
    fi

    current_restart_count="$(container_restart_count "${running_container_id}")"
    restart_delta=$(( current_restart_count - baseline_restart_count ))
    if (( restart_delta >= POST_DEPLOY_CRASH_THRESHOLD )); then
      ATTEMPT_REASON="crash_loop_detected"
      ATTEMPT_OUTCOME="post_deploy_crash_loop"
      break
    fi

    sleep "${UPDATE_POLL_SECONDS}"
  done

  if [[ "${ATTEMPT_OUTCOME}" == "post_deploy_monitor_failed" || "${ATTEMPT_OUTCOME}" == "post_deploy_crash_loop" ]]; then
    if rollback_to_last_known_good "${ATTEMPT_REASON}" "${pre_alembic_version}" "${backup_path}"; then
      set_updates_paused "post_deploy_crash_loop"
      ATTEMPT_OUTCOME="rolled_back_after_crash_loop"
      finish_attempt 1
      return $?
    fi
    set_updates_paused "rollback_failed_after_post_deploy_crash_loop"
    ATTEMPT_OUTCOME="catastrophic_rollback_failed"
    ATTEMPT_REASON="rollback_failed_after_post_deploy_crash_loop"
    finish_attempt 1
    return $?
  fi

  ATTEMPT_OUTCOME="success"
  ATTEMPT_REASON="update_applied_and_healthy"
  finish_attempt 0
  return $?
}

run_manual_rollback() {
  local reason="$1"
  local pre_alembic_version

  ATTEMPT_STARTED_AT="$(now_utc_iso)"
  ATTEMPT_STARTED_EPOCH="$(date +%s)"
  ATTEMPT_CURRENT_VERSION="${CURRENT_SHA}"
  ATTEMPT_TARGET_VERSION="${LKG_SHA}"
  ATTEMPT_BACKUP_PATH="${LAST_BACKUP_PATH}"
  ATTEMPT_ROLLBACK="true"
  ATTEMPT_REASON="${reason}"
  ATTEMPT_OUTCOME="manual_rollback_failed"

  pre_alembic_version="$(get_alembic_version 2>/dev/null || true)"
  if rollback_to_last_known_good "${reason}" "${pre_alembic_version}" "${LAST_BACKUP_PATH}"; then
    ATTEMPT_OUTCOME="manual_rollback_succeeded"
    finish_attempt 0
    return $?
  fi

  set_updates_paused "manual_rollback_failed"
  ATTEMPT_OUTCOME="catastrophic_rollback_failed"
  finish_attempt 1
  return $?
}

resume_updates() {
  UPDATES_PAUSED="false"
  PAUSE_REASON=""
  LAST_ROLLBACK_REASON=""
  write_metadata
  log_json "INFO" "updates_resumed" "operator" "${USER:-unknown}"
}

print_status() {
  cat <<EOF_STATUS
auto_update_enabled=${AUTO_UPDATE_ENABLED}
updates_paused=${UPDATES_PAUSED}
pause_reason=${PAUSE_REASON}
current_sha=${CURRENT_SHA}
lkg_sha=${LKG_SHA}
last_update_attempt_at=${LAST_UPDATE_ATTEMPT_AT}
last_update_result=${LAST_UPDATE_RESULT}
last_target_sha=${LAST_TARGET_SHA}
last_backup_path=${LAST_BACKUP_PATH}
last_duration_seconds=${LAST_DURATION_SECONDS}
last_rollback_reason=${LAST_ROLLBACK_REASON}
last_scheduled_run_date=${LAST_SCHEDULED_RUN_DATE}
EOF_STATUS
}

init() {
  cd "${REPO_ROOT}"

  load_env_file "${ENV_FILE}"
  apply_defaults
  ensure_runtime_paths
  load_metadata

  validate_numeric_settings
}

main() {
  local command run_mode rollback_reason
  command="${1:-run}"

  init

  case "${command}" in
    run)
      run_mode="${2:---scheduled}"
      require_run_dependencies
      if ! acquire_lock; then
        log_json "INFO" "skipped_locked" "command" "run"
        exit 0
      fi
      if [[ "${run_mode}" == "--scheduled" ]]; then
        run_update "scheduled"
      elif [[ "${run_mode}" == "--force" ]]; then
        run_update "force"
      else
        usage
        exit 2
      fi
      ;;
    rollback)
      rollback_reason="manual_rollback"
      shift || true
      require_rollback_dependencies
      while (( $# > 0 )); do
        case "$1" in
          --reason)
            rollback_reason="${2:-manual_rollback}"
            shift 2
            ;;
          *)
            usage
            exit 2
            ;;
        esac
      done
      if ! acquire_lock; then
        log_json "INFO" "skipped_locked" "command" "rollback"
        exit 1
      fi
      run_manual_rollback "${rollback_reason}"
      ;;
    resume-updates)
      require_resume_dependencies
      if ! acquire_lock; then
        log_json "INFO" "skipped_locked" "command" "resume-updates"
        exit 1
      fi
      resume_updates
      ;;
    status)
      print_status
      ;;
    -h|--help|help)
      usage
      ;;
    *)
      usage
      exit 2
      ;;
  esac
}

trap release_lock EXIT

main "$@"
