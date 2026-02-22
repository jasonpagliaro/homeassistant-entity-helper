#!/usr/bin/env bash
set -euo pipefail

REPO_SLUG="${REPO_SLUG:-}"
APPROVAL_COUNT="${APPROVAL_COUNT:-1}"
DRY_RUN="false"

STATUS_CHECKS=("build")

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
  cat <<EOF_USAGE
Usage: $(basename "$0") [options]

Configure GitHub branch protection for main and enforce squash-only merges.
Requires repository admin permission and GitHub CLI authentication.

Options:
  --repo owner/name       Repository slug (default: auto-detect from origin remote)
  --approvals N           Required PR approvals on main (default: ${APPROVAL_COUNT})
  --dry-run               Print API payloads without applying
  -h, --help              Show this help message
EOF_USAGE
}

require_command() {
  local command_name="$1"
  command -v "${command_name}" >/dev/null 2>&1 || die 2 "Missing required command: ${command_name}"
}

detect_repo_slug() {
  local remote_url="$1"
  local slug=""

  if [[ "${remote_url}" =~ ^https://github.com/([^/]+/[^/]+)(\.git)?$ ]]; then
    slug="${BASH_REMATCH[1]}"
  elif [[ "${remote_url}" =~ ^git@github.com:([^/]+/[^/]+)(\.git)?$ ]]; then
    slug="${BASH_REMATCH[1]}"
  fi

  slug="${slug%.git}"
  [[ -n "${slug}" ]] || die 3 "Unable to parse GitHub repo slug from origin URL: ${remote_url}"
  printf "%s\n" "${slug}"
}

json_status_checks() {
  local out="["
  local item
  for item in "${STATUS_CHECKS[@]}"; do
    out+="\"${item}\","
  done
  out="${out%,}]"
  printf "%s\n" "${out}"
}

while (( "$#" )); do
  case "$1" in
    --repo)
      REPO_SLUG="$2"
      shift 2
      ;;
    --approvals)
      APPROVAL_COUNT="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN="true"
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

require_command git
require_command gh

if [[ -z "${REPO_SLUG}" ]]; then
  REPO_SLUG="$(detect_repo_slug "$(git remote get-url origin)")"
fi

if [[ ! "${APPROVAL_COUNT}" =~ ^[0-9]+$ ]]; then
  die 4 "--approvals must be a non-negative integer."
fi

STATUS_CHECKS_JSON="$(json_status_checks)"
PROTECTION_PAYLOAD="$(cat <<JSON
{
  "required_status_checks": {
    "strict": true,
    "contexts": ${STATUS_CHECKS_JSON}
  },
  "enforce_admins": false,
  "required_pull_request_reviews": {
    "dismiss_stale_reviews": true,
    "require_code_owner_reviews": false,
    "required_approving_review_count": ${APPROVAL_COUNT},
    "require_last_push_approval": false
  },
  "restrictions": null,
  "required_linear_history": false,
  "allow_force_pushes": false,
  "allow_deletions": false,
  "block_creations": false,
  "required_conversation_resolution": true,
  "lock_branch": false,
  "allow_fork_syncing": true
}
JSON
)"

if [[ "${DRY_RUN}" == "true" ]]; then
  cat <<EOF_DRY_RUN
Repo: ${REPO_SLUG}

Branch protection payload:
${PROTECTION_PAYLOAD}

Repository merge settings:
allow_squash_merge=true
allow_merge_commit=false
allow_rebase_merge=false
EOF_DRY_RUN
  exit 0
fi

gh auth status >/dev/null

gh api \
  --method PUT \
  "repos/${REPO_SLUG}/branches/main/protection" \
  --input - <<<"${PROTECTION_PAYLOAD}" >/dev/null

gh api \
  --method PATCH \
  "repos/${REPO_SLUG}" \
  -f allow_squash_merge=true \
  -f allow_merge_commit=false \
  -f allow_rebase_merge=false >/dev/null

cat <<EOF_DONE
GitHub main protection configured for ${REPO_SLUG}.
- PR required on main
- Required checks: $(IFS=,; echo "${STATUS_CHECKS[*]}")
- Squash merge enabled
- Merge commit and rebase merge disabled
EOF_DONE
