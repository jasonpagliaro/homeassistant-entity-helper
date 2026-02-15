#!/usr/bin/env bash
set -euo pipefail

SCRIPT_PATH="$(realpath "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_REPO_PATH="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEFAULT_ENV_FILE="/etc/default/ha-entity-vault-updater"
REMOTE_NAME="origin"
APP_DIR_IN_CONTAINER="/srv/ha-entity-vault"
SERVICE_NAME="ha-entity-vault.service"

if [[ -f "${HEV_UPDATE_ENV_FILE:-${DEFAULT_ENV_FILE}}" ]]; then
  # shellcheck disable=SC1090
  source "${HEV_UPDATE_ENV_FILE:-${DEFAULT_ENV_FILE}}"
fi

HEV_UPDATE_MODE="${HEV_UPDATE_MODE:-detect_approve}"
HEV_UPDATE_CONTAINER="${HEV_UPDATE_CONTAINER:-ha-entity-vault}"
HEV_UPDATE_REPO_PATH="${HEV_UPDATE_REPO_PATH:-${DEFAULT_REPO_PATH}}"
HEV_UPDATE_BRANCH="${HEV_UPDATE_BRANCH:-main}"
HEV_UPDATE_STATE_DIR="${HEV_UPDATE_STATE_DIR:-/var/lib/ha-entity-vault-updater}"
HEV_UPDATE_BACKUP_RETENTION="${HEV_UPDATE_BACKUP_RETENTION:-14}"
HEV_UPDATE_ALLOW_HIGH_RISK_AUTO="${HEV_UPDATE_ALLOW_HIGH_RISK_AUTO:-false}"
HEV_UPDATE_LOCK_FILE="${HEV_UPDATE_LOCK_FILE:-/var/lock/ha-entity-vault-updater.lock}"
HEV_UPDATE_HEALTH_TIMEOUT_SEC="${HEV_UPDATE_HEALTH_TIMEOUT_SEC:-5}"

PENDING_FILE="${HEV_UPDATE_STATE_DIR}/pending-update.json"

LOCAL_SHA=""
REMOTE_SHA=""
COMMITS_AHEAD="0"
CHECK_RESULT="none"
LOCAL_AHEAD="false"
HIGH_RISK="false"
CHANGED_PATHS=()
HIGH_RISK_REASONS=()

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

is_positive_integer() {
  local value="$1"
  [[ "${value}" =~ ^[0-9]+$ ]] && (( value > 0 ))
}

normalize_bool() {
  case "${1,,}" in
    1|true|yes|on)
      printf "true\n"
      ;;
    0|false|no|off)
      printf "false\n"
      ;;
    *)
      printf "false\n"
      ;;
  esac
}

join_by() {
  local delimiter="$1"
  shift || true
  local first="true"
  local item
  for item in "$@"; do
    if [[ "${first}" == "true" ]]; then
      first="false"
      printf "%s" "${item}"
    else
      printf "%s%s" "${delimiter}" "${item}"
    fi
  done
}

validate_mode() {
  case "${HEV_UPDATE_MODE}" in
    check_only|detect_approve|auto_apply)
      ;;
    *)
      die 2 "Invalid HEV_UPDATE_MODE '${HEV_UPDATE_MODE}'. Expected check_only, detect_approve, or auto_apply."
      ;;
  esac
}

require_command() {
  local command_name="$1"
  if ! command -v "${command_name}" >/dev/null 2>&1; then
    die 3 "Missing required command: ${command_name}"
  fi
}

git_repo() {
  git -c safe.directory="${HEV_UPDATE_REPO_PATH}" -C "${HEV_UPDATE_REPO_PATH}" "$@"
}

acquire_lock() {
  mkdir -p "$(dirname "${HEV_UPDATE_LOCK_FILE}")"
  exec 9>"${HEV_UPDATE_LOCK_FILE}"
  if ! flock -n 9; then
    die 98 "Another updater instance is running (lock: ${HEV_UPDATE_LOCK_FILE})."
  fi
}

ensure_prereqs() {
  validate_mode
  require_command git
  require_command lxc
  require_command flock

  HEV_UPDATE_ALLOW_HIGH_RISK_AUTO="$(normalize_bool "${HEV_UPDATE_ALLOW_HIGH_RISK_AUTO}")"

  if ! is_positive_integer "${HEV_UPDATE_BACKUP_RETENTION}"; then
    die 4 "HEV_UPDATE_BACKUP_RETENTION must be a positive integer."
  fi
  if ! is_positive_integer "${HEV_UPDATE_HEALTH_TIMEOUT_SEC}"; then
    die 5 "HEV_UPDATE_HEALTH_TIMEOUT_SEC must be a positive integer."
  fi

  if [[ ! -d "${HEV_UPDATE_REPO_PATH}" ]]; then
    die 6 "HEV_UPDATE_REPO_PATH does not exist: ${HEV_UPDATE_REPO_PATH}"
  fi
  if [[ ! -d "${HEV_UPDATE_REPO_PATH}/.git" ]]; then
    die 7 "HEV_UPDATE_REPO_PATH is not a git repository: ${HEV_UPDATE_REPO_PATH}"
  fi
  if [[ ! -f "${HEV_UPDATE_REPO_PATH}/lxc/update-in-container.sh" ]]; then
    die 8 "Missing in-container helper at ${HEV_UPDATE_REPO_PATH}/lxc/update-in-container.sh"
  fi

  git_repo remote get-url "${REMOTE_NAME}" >/dev/null 2>&1 || die 9 "Git remote '${REMOTE_NAME}' is not configured."

  lxc info "${HEV_UPDATE_CONTAINER}" >/dev/null 2>&1 || die 10 "LXC container '${HEV_UPDATE_CONTAINER}' not found."

  mkdir -p "${HEV_UPDATE_STATE_DIR}"
}

fetch_remote() {
  git_repo fetch --quiet "${REMOTE_NAME}" "${HEV_UPDATE_BRANCH}" || die 11 "Failed to fetch ${REMOTE_NAME}/${HEV_UPDATE_BRANCH}."
  git_repo rev-parse --verify --quiet "refs/remotes/${REMOTE_NAME}/${HEV_UPDATE_BRANCH}" >/dev/null || \
    die 12 "Remote branch ${REMOTE_NAME}/${HEV_UPDATE_BRANCH} not found."
}

add_high_risk_reason() {
  local reason="$1"
  local existing
  for existing in "${HIGH_RISK_REASONS[@]}"; do
    if [[ "${existing}" == "${reason}" ]]; then
      return
    fi
  done
  HIGH_RISK_REASONS+=("${reason}")
}

detect_high_risk() {
  HIGH_RISK="false"
  HIGH_RISK_REASONS=()

  local path
  for path in "${CHANGED_PATHS[@]}"; do
    case "${path}" in
      requirements.txt)
        add_high_risk_reason "requirements_changed"
        ;;
      migrations/*)
        add_high_risk_reason "migrations_changed"
        ;;
      lxc/ha-entity-vault.service)
        add_high_risk_reason "service_unit_changed"
        ;;
    esac
  done

  if (( ${#HIGH_RISK_REASONS[@]} > 0 )); then
    HIGH_RISK="true"
  fi
}

evaluate_git_state() {
  fetch_remote

  LOCAL_SHA="$(git_repo rev-parse HEAD)"
  REMOTE_SHA="$(git_repo rev-parse "refs/remotes/${REMOTE_NAME}/${HEV_UPDATE_BRANCH}")"
  COMMITS_AHEAD="0"
  CHECK_RESULT="none"
  LOCAL_AHEAD="false"
  CHANGED_PATHS=()
  HIGH_RISK="false"
  HIGH_RISK_REASONS=()

  if [[ "${LOCAL_SHA}" == "${REMOTE_SHA}" ]]; then
    return
  fi

  if git_repo merge-base --is-ancestor "${LOCAL_SHA}" "${REMOTE_SHA}"; then
    CHECK_RESULT="updates"
    COMMITS_AHEAD="$(git_repo rev-list --count "${LOCAL_SHA}..${REMOTE_SHA}")"
    mapfile -t CHANGED_PATHS < <(git_repo diff --name-only "${LOCAL_SHA}" "${REMOTE_SHA}")
    detect_high_risk
    return
  fi

  if git_repo merge-base --is-ancestor "${REMOTE_SHA}" "${LOCAL_SHA}"; then
    LOCAL_AHEAD="true"
    log "WARN" "Local branch is ahead of ${REMOTE_NAME}/${HEV_UPDATE_BRANCH}; no fast-forward update available."
    return
  fi

  die 13 "Local branch has diverged from ${REMOTE_NAME}/${HEV_UPDATE_BRANCH}; manual reconciliation required."
}

json_reasons_array() {
  if (( ${#HIGH_RISK_REASONS[@]} == 0 )); then
    printf "[]\n"
    return
  fi

  local out="["
  local reason
  for reason in "${HIGH_RISK_REASONS[@]}"; do
    out+="\"${reason}\","
  done
  out="${out%,}]"
  printf "%s\n" "${out}"
}

write_pending_state() {
  local reasons_json
  reasons_json="$(json_reasons_array)"

  cat > "${PENDING_FILE}" <<EOF
{
  "created_at": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "branch": "${HEV_UPDATE_BRANCH}",
  "local_sha": "${LOCAL_SHA}",
  "target_sha": "${REMOTE_SHA}",
  "commits_ahead": ${COMMITS_AHEAD},
  "high_risk": ${HIGH_RISK},
  "high_risk_reasons": ${reasons_json},
  "mode": "detect_approve"
}
EOF
}

clear_pending_state() {
  if [[ -f "${PENDING_FILE}" ]]; then
    rm -f "${PENDING_FILE}"
  fi
}

ensure_clean_worktree() {
  if [[ -n "$(git_repo status --porcelain --untracked-files=normal)" ]]; then
    die 14 "Repository has uncommitted changes; refusing update. Commit/stash/clean first."
  fi
}

run_container_helper() {
  local helper_command="$1"
  shift || true
  lxc exec "${HEV_UPDATE_CONTAINER}" -- \
    env HEV_UPDATE_BACKUP_RETENTION="${HEV_UPDATE_BACKUP_RETENTION}" HEV_UPDATE_HEALTH_TIMEOUT_SEC="${HEV_UPDATE_HEALTH_TIMEOUT_SEC}" \
    bash "${APP_DIR_IN_CONTAINER}/lxc/update-in-container.sh" "${helper_command}" "$@"
}

restart_service() {
  lxc exec "${HEV_UPDATE_CONTAINER}" -- systemctl restart "${SERVICE_NAME}"
}

rollback_after_failure() {
  local pre_sha="$1"
  local backup_path="$2"

  log "ERROR" "Update failed post-pull; starting rollback to ${pre_sha}."

  if ! git_repo reset --hard "${pre_sha}" >/dev/null; then
    die 31 "Rollback failed while resetting git state to ${pre_sha}."
  fi
  if ! run_container_helper install; then
    die 31 "Rollback failed while reinstalling dependencies."
  fi
  if ! run_container_helper restore-db "${backup_path}"; then
    die 31 "Rollback failed while restoring database backup ${backup_path}."
  fi
  if ! restart_service; then
    die 31 "Rollback failed while restarting ${SERVICE_NAME}."
  fi
  if ! run_container_helper smoke; then
    die 31 "Rollback failed smoke verification; manual intervention required."
  fi

  log "WARN" "Rollback completed successfully. Update was not applied."
  return 30
}

apply_update() {
  ensure_clean_worktree
  evaluate_git_state

  if [[ "${CHECK_RESULT}" != "updates" ]]; then
    clear_pending_state
    log "INFO" "No remote updates to apply (${LOCAL_SHA})."
    return 0
  fi

  local pre_sha="${LOCAL_SHA}"
  local target_sha="${REMOTE_SHA}"
  log "INFO" "Applying update ${pre_sha} -> ${target_sha} on branch ${HEV_UPDATE_BRANCH}."

  local backup_path
  backup_path="$(run_container_helper backup-db | tail -n1)"
  if [[ -z "${backup_path}" ]]; then
    die 15 "Database backup did not return a path."
  fi
  log "INFO" "Pre-update database backup: ${backup_path}"

  if ! git_repo pull --ff-only "${REMOTE_NAME}" "${HEV_UPDATE_BRANCH}"; then
    die 16 "git pull --ff-only failed."
  fi

  if ! run_container_helper install; then
    rollback_after_failure "${pre_sha}" "${backup_path}" || return $?
    return 30
  fi
  if ! restart_service; then
    rollback_after_failure "${pre_sha}" "${backup_path}" || return $?
    return 30
  fi
  if ! run_container_helper smoke; then
    rollback_after_failure "${pre_sha}" "${backup_path}" || return $?
    return 30
  fi

  clear_pending_state
  log "INFO" "Update applied successfully at ${target_sha}."
  return 0
}

print_status() {
  evaluate_git_state

  printf "mode=%s\n" "${HEV_UPDATE_MODE}"
  printf "container=%s\n" "${HEV_UPDATE_CONTAINER}"
  printf "repo=%s\n" "${HEV_UPDATE_REPO_PATH}"
  printf "branch=%s\n" "${HEV_UPDATE_BRANCH}"
  printf "local_sha=%s\n" "${LOCAL_SHA}"
  printf "remote_sha=%s\n" "${REMOTE_SHA}"
  printf "update_available=%s\n" "$([[ "${CHECK_RESULT}" == "updates" ]] && echo "true" || echo "false")"
  printf "commits_ahead=%s\n" "${COMMITS_AHEAD}"
  printf "local_ahead=%s\n" "${LOCAL_AHEAD}"
  printf "high_risk=%s\n" "${HIGH_RISK}"
  printf "high_risk_reasons=%s\n" "$(join_by "," "${HIGH_RISK_REASONS[@]}")"
  printf "pending_state=%s\n" "$([[ -f "${PENDING_FILE}" ]] && echo "present" || echo "absent")"
  printf "pending_file=%s\n" "${PENDING_FILE}"
}

cmd_check() {
  evaluate_git_state
  if [[ "${CHECK_RESULT}" == "updates" ]]; then
    log "INFO" "Update available: ${COMMITS_AHEAD} commit(s), ${LOCAL_SHA} -> ${REMOTE_SHA}."
    if [[ "${HIGH_RISK}" == "true" ]]; then
      log "WARN" "High-risk changes detected: $(join_by "," "${HIGH_RISK_REASONS[@]}")."
    fi
    exit 10
  fi
  log "INFO" "No updates available."
}

cmd_apply() {
  apply_update
}

cmd_run() {
  case "${HEV_UPDATE_MODE}" in
    check_only)
      evaluate_git_state
      if [[ "${CHECK_RESULT}" == "updates" ]]; then
        log "INFO" "check_only mode: updates available (${COMMITS_AHEAD} commit(s), ${LOCAL_SHA} -> ${REMOTE_SHA})."
      else
        log "INFO" "check_only mode: no updates available."
      fi
      ;;
    detect_approve)
      evaluate_git_state
      if [[ "${CHECK_RESULT}" == "updates" ]]; then
        write_pending_state
        log "INFO" "detect_approve mode: pending update saved to ${PENDING_FILE}."
        log "INFO" "Approve/apply manually with: sudo ${SCRIPT_PATH} apply"
      else
        clear_pending_state
        log "INFO" "detect_approve mode: no updates available."
      fi
      ;;
    auto_apply)
      evaluate_git_state
      if [[ "${CHECK_RESULT}" != "updates" ]]; then
        clear_pending_state
        log "INFO" "auto_apply mode: no updates available."
        return 0
      fi

      if [[ "${HIGH_RISK}" == "true" && "${HEV_UPDATE_ALLOW_HIGH_RISK_AUTO}" != "true" ]]; then
        write_pending_state
        die 17 "auto_apply blocked by high-risk changes: $(join_by "," "${HIGH_RISK_REASONS[@]}"). Manual apply required."
      fi

      apply_update
      ;;
  esac
}

usage() {
  cat <<EOF
Usage: $(basename "$0") <check|apply|run|status>

Commands:
  check   Fetch remote and detect updates (exit 10 when updates are available)
  apply   Apply updates with smoke checks and rollback on failure
  run     Execute configured automation mode (HEV_UPDATE_MODE)
  status  Print current mode, commit state, and pending-state location
EOF
}

main() {
  local command="${1:-}"
  case "${command}" in
    check|apply|run|status)
      ;;
    *)
      usage
      exit 64
      ;;
  esac

  acquire_lock
  ensure_prereqs

  case "${command}" in
    check)
      cmd_check
      ;;
    apply)
      cmd_apply
      ;;
    run)
      cmd_run
      ;;
    status)
      print_status
      ;;
  esac
}

main "$@"
