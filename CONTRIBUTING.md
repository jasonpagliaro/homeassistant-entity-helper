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

## Pull requests
1. Keep PRs focused and include tests for behavior changes.
2. Do not commit real Home Assistant tokens or private env files.
3. Update docs (`README.md`) when adding features or endpoints.
4. Ensure CI is green before requesting review.

## Code style
- Keep modules small and testable.
- Prefer explicit data flow over implicit globals.
- Avoid logging secrets (especially tokens).
