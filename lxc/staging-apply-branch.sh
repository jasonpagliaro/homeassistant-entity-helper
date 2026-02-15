#!/usr/bin/env bash
set -euo pipefail

BRANCH="${BRANCH:-}"
STAGING_REPO_PATH="${STAGING_REPO_PATH:-/srv/ha-entity-vault-staging}"
STAGING_CONTAINER="${STAGING_CONTAINER:-ha-entity-vault-staging}"
STATE_DIR="${STATE_DIR:-/tmp/ha-entity-vault-staging-updater}"
LOCK_FILE="${LOCK_FILE:-}"

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
Usage: $(basename "$0") --branch BRANCH [options]

Fetch/switch staging checkout to a feature branch and apply it to the staging container.

Options:
  --branch BRANCH            Branch to validate on staging (required)
  --staging-path PATH        Staging host checkout path (default: ${STAGING_REPO_PATH})
  --staging-container NAME   Staging LXC container name (default: ${STAGING_CONTAINER})
  --state-dir PATH           Override updater state dir (default: ${STATE_DIR})
  --lock-file PATH           Override updater lock file (default: <state-dir>/update.lock)
  -h, --help                 Show this help message
EOF
}

require_command() {
  local command_name="$1"
  command -v "${command_name}" >/dev/null 2>&1 || die 2 "Missing required command: ${command_name}"
}

while (( "$#" )); do
  case "$1" in
    --branch)
      BRANCH="$2"
      shift 2
      ;;
    --staging-path)
      STAGING_REPO_PATH="$2"
      shift 2
      ;;
    --staging-container)
      STAGING_CONTAINER="$2"
      shift 2
      ;;
    --state-dir)
      STATE_DIR="$2"
      shift 2
      ;;
    --lock-file)
      LOCK_FILE="$2"
      shift 2
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

[[ -n "${BRANCH}" ]] || die 64 "--branch is required."
require_command git
require_command lxc

if [[ ! -d "${STAGING_REPO_PATH}/.git" ]]; then
  die 3 "Staging checkout is missing or not a git repo: ${STAGING_REPO_PATH}"
fi
if [[ ! -x "${STAGING_REPO_PATH}/lxc/update-host.sh" ]]; then
  die 4 "Missing executable updater: ${STAGING_REPO_PATH}/lxc/update-host.sh"
fi

if [[ -n "$(git -C "${STAGING_REPO_PATH}" status --porcelain --untracked-files=normal)" ]]; then
  die 5 "Staging checkout has uncommitted changes; clean it before switching branches."
fi

git -C "${STAGING_REPO_PATH}" fetch --prune origin
git -C "${STAGING_REPO_PATH}" rev-parse --verify --quiet "refs/remotes/origin/${BRANCH}" >/dev/null || \
  die 6 "Branch not found on origin: ${BRANCH}"

if git -C "${STAGING_REPO_PATH}" rev-parse --verify --quiet "refs/heads/${BRANCH}" >/dev/null; then
  git -C "${STAGING_REPO_PATH}" switch "${BRANCH}"
else
  git -C "${STAGING_REPO_PATH}" switch -c "${BRANCH}" --track "origin/${BRANCH}"
fi

git -C "${STAGING_REPO_PATH}" pull --ff-only origin "${BRANCH}"

if [[ -z "${LOCK_FILE}" ]]; then
  LOCK_FILE="${STATE_DIR}/update.lock"
fi
mkdir -p "${STATE_DIR}"

log "INFO" "Applying ${BRANCH} to staging container ${STAGING_CONTAINER}."
HEV_UPDATE_ENV_FILE=/dev/null \
HEV_UPDATE_CONTAINER="${STAGING_CONTAINER}" \
HEV_UPDATE_REPO_PATH="${STAGING_REPO_PATH}" \
HEV_UPDATE_BRANCH="${BRANCH}" \
HEV_UPDATE_STATE_DIR="${STATE_DIR}" \
HEV_UPDATE_LOCK_FILE="${LOCK_FILE}" \
"${STAGING_REPO_PATH}/lxc/update-host.sh" apply

cat <<EOF

Staging apply complete.
Branch:    ${BRANCH}
Checkout:  ${STAGING_REPO_PATH}
Container: ${STAGING_CONTAINER}

Next:
1. Validate feature behavior in staging UI/API.
2. Add staging validation notes to your PR before merge.
EOF
