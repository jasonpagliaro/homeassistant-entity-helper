# Contributing

## Development setup
1. Install Python 3.12.
2. Install dependencies: `make dev-install`.
3. Start app locally: `make run`.

## LXC deployment scripts
- Host helper: `lxc/create-lxc-container.sh`
- In-container bootstrap: `lxc/setup-in-container.sh`
- Service unit: `lxc/ha-entity-vault.service`
- Host updater: `lxc/update-host.sh`
- In-container updater helper: `lxc/update-in-container.sh`
- Host workflow bootstrap: `lxc/bootstrap-host-workflow.sh`
- Staging branch apply helper: `lxc/staging-apply-branch.sh`
- Production auto-apply setup helper: `lxc/configure-prod-auto-apply.sh`
- GitHub branch protection helper: `lxc/configure-github-main-protection.sh`
- Update timer/service templates:
  - `lxc/ha-entity-vault-update-check.service`
  - `lxc/ha-entity-vault-update-check.timer`

## Quality checks
- Lint: `make lint`
- Typecheck: `make typecheck`
- Tests: `make test`
- Shell syntax:
  - `bash -n lxc/create-lxc-container.sh`
  - `bash -n lxc/setup-in-container.sh`
  - `bash -n lxc/update-host.sh`
  - `bash -n lxc/update-in-container.sh`
  - `bash -n lxc/bootstrap-host-workflow.sh`
  - `bash -n lxc/staging-apply-branch.sh`
  - `bash -n lxc/configure-prod-auto-apply.sh`
  - `bash -n lxc/configure-github-main-protection.sh`

## Pull requests
1. Keep PRs focused and include tests for behavior changes.
2. Do not commit real Home Assistant tokens or private env files.
3. Update docs (`README.md`) when adding features or endpoints.
4. Ensure CI is green before requesting review.
5. For behavior changes, validate branch on staging before merging to `main`.

## Code style
- Keep modules small and testable.
- Prefer explicit data flow over implicit globals.
- Avoid logging secrets (especially tokens).
