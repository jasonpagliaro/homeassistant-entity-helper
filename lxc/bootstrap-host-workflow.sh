#!/usr/bin/env bash
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/jasonpagliaro/homeassistant-entity-helper.git}"
PROD_REPO_PATH="${PROD_REPO_PATH:-/srv/ha-entity-vault-prod}"
STAGING_REPO_PATH="${STAGING_REPO_PATH:-/srv/ha-entity-vault-staging}"
STAGING_DATA_PATH="${STAGING_DATA_PATH:-/srv/ha-entity-vault-data-staging}"
STAGING_CONTAINER="${STAGING_CONTAINER:-ha-entity-vault-staging}"
STAGING_PORT="${STAGING_PORT:-18001}"

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

Bootstrap production + staging host checkouts and create/update a staging LXC container.

Options:
  --repo-url URL             Git clone URL (default: ${REPO_URL})
  --prod-path PATH           Production host checkout path (default: ${PROD_REPO_PATH})
  --staging-path PATH        Staging host checkout path (default: ${STAGING_REPO_PATH})
  --staging-data-path PATH   Staging data path (default: ${STAGING_DATA_PATH})
  --staging-container NAME   Staging LXC container name (default: ${STAGING_CONTAINER})
  --staging-port PORT        Staging host port (default: ${STAGING_PORT})
  -h, --help                 Show this help message
EOF
}

require_command() {
  local command_name="$1"
  command -v "${command_name}" >/dev/null 2>&1 || die 2 "Missing required command: ${command_name}"
}

clone_or_validate_checkout() {
  local path="$1"
  local role="$2"

  if [[ -d "${path}/.git" ]]; then
    local remote_url
    remote_url="$(git -C "${path}" remote get-url origin 2>/dev/null || true)"
    if [[ -n "${remote_url}" && "${remote_url}" != "${REPO_URL}" ]]; then
      log "WARN" "${role} checkout remote differs from --repo-url (${remote_url} != ${REPO_URL})."
    fi
    git -C "${path}" fetch --prune origin
    return
  fi

  if [[ -e "${path}" ]]; then
    if [[ -d "${path}" && -z "$(ls -A "${path}" 2>/dev/null)" ]]; then
      :
    else
      die 3 "${role} path exists and is not a git checkout: ${path}"
    fi
  fi

  mkdir -p "$(dirname "${path}")"
  git clone "${REPO_URL}" "${path}"
}

while (( "$#" )); do
  case "$1" in
    --repo-url)
      REPO_URL="$2"
      shift 2
      ;;
    --prod-path)
      PROD_REPO_PATH="$2"
      shift 2
      ;;
    --staging-path)
      STAGING_REPO_PATH="$2"
      shift 2
      ;;
    --staging-data-path)
      STAGING_DATA_PATH="$2"
      shift 2
      ;;
    --staging-container)
      STAGING_CONTAINER="$2"
      shift 2
      ;;
    --staging-port)
      STAGING_PORT="$2"
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

require_command git
require_command lxc

log "INFO" "Preparing production checkout: ${PROD_REPO_PATH}"
clone_or_validate_checkout "${PROD_REPO_PATH}" "Production"

log "INFO" "Preparing staging checkout: ${STAGING_REPO_PATH}"
clone_or_validate_checkout "${STAGING_REPO_PATH}" "Staging"

mkdir -p "${STAGING_DATA_PATH}"

if [[ ! -x "${STAGING_REPO_PATH}/lxc/create-lxc-container.sh" ]]; then
  die 4 "Missing executable script: ${STAGING_REPO_PATH}/lxc/create-lxc-container.sh"
fi

log "INFO" "Creating/updating staging container '${STAGING_CONTAINER}' on port ${STAGING_PORT}."
"${STAGING_REPO_PATH}/lxc/create-lxc-container.sh" \
  "${STAGING_CONTAINER}" \
  "${STAGING_REPO_PATH}" \
  "${STAGING_DATA_PATH}" \
  "${STAGING_PORT}"

cat <<EOF

Bootstrap complete.

Production checkout: ${PROD_REPO_PATH}
Staging checkout:    ${STAGING_REPO_PATH}
Staging container:   ${STAGING_CONTAINER}
Staging URL:         http://<host-ip>:${STAGING_PORT}

Next:
1. Configure production auto-apply:
   sudo ${PROD_REPO_PATH}/lxc/configure-prod-auto-apply.sh --prod-path ${PROD_REPO_PATH}
2. Configure GitHub branch protection + squash-only merges:
   ${PROD_REPO_PATH}/lxc/configure-github-main-protection.sh
EOF
