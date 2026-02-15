# HA Entity Vault

HA Entity Vault is a self-hosted single-container app for pulling Home Assistant entities on demand, browsing/filtering them, and exporting filtered views as JSON/CSV.

- Stack: Python 3.12, FastAPI, SQLModel + SQLite, Jinja2, httpx, Alembic.
- Container runtime target: Linux Containers (LXC).
- Data persistence: host-mounted `/data` volume inside the LXC container.
- UI name is configurable via `APP_NAME`, so repo/product naming is easy to change later.

## MVP Features
- Multiple Home Assistant profiles.
- Settings per profile:
  - Base URL
  - Long-lived access token
  - Optional token env var override name (`token_env_var`, default `HA_TOKEN`)
  - Verify TLS toggle
  - Request timeout seconds
- Test connection (`GET /api/config`) with actionable status.
- Sync now (`GET /api/states`) and store entity snapshots.
- Registry enrichment during sync (best effort):
  - entity registry
  - device registry
  - area registry
  - label registry
  - floor registry
- Entity table with:
  - Search (`entity_id`, friendly name, area/location, labels)
  - Filters (domain/state/changed recently)
  - Server-side pagination
- Entity detail page with enriched metadata plus prettified attributes/context JSON.
- Export filtered results as JSON or CSV (includes `pulled_at` and profile context).
- Health endpoint: `GET /healthz`.
- Structured JSON logs with request IDs and sync timing/count fields.

## Data Model (Snapshot Approach)
MVP uses immutable snapshot runs:

1. `profiles`
2. `sync_runs`
3. `entity_snapshots`

Each sync creates one `sync_runs` row and N `entity_snapshots` rows linked by `sync_run_id`, preserving point-in-time views for future diffing/history features.

## LXC Quick Start (Recommended)
These commands create an LXC container through LXD (Ubuntu 24.04), bind-mount this repo into `/srv/ha-entity-vault`, mount host data into `/data`, then install and start the systemd service.

From this repo root:

```bash
./lxc/create-lxc-container.sh ha-entity-vault "$(pwd)" "$HOME/ha-entity-vault-data" 18000
```

The script:
- launches/updates container `ha-entity-vault`
- mounts repo and data with shifted id mapping
- installs app dependencies inside container
- enables `ha-entity-vault.service`
- exposes app on host port `18000` via LXD proxy device

Open:

- App UI: `http://<host-ip>:18000`
- OpenAPI docs: `http://<host-ip>:18000/docs`

### macOS note
LXC containers require a Linux kernel. On macOS, use a Linux host/VM and run the same `lxc/create-lxc-container.sh` there.

## LXC Layout
- App code mount: `/srv/ha-entity-vault`
- Persistent data mount: `/data`
- Virtualenv: `/opt/ha-entity-vault/.venv`
- Service unit: `/etc/systemd/system/ha-entity-vault.service`
- Service env file: `/etc/default/ha-entity-vault`

## LXC Update Workflow
The project includes a host-orchestrated app updater with three automation levels:

- `check_only`: only check for updates and log findings.
- `detect_approve` (default): detect updates, write pending metadata, wait for explicit manual apply.
- `auto_apply`: apply updates automatically unless blocked by safety gates.

Update scripts:
- Host orchestrator: `lxc/update-host.sh`
- In-container helper: `lxc/update-in-container.sh`
- Update config example: `lxc/ha-entity-vault-update.env.example`
- Timer/service templates:
  - `lxc/ha-entity-vault-update-check.service`
  - `lxc/ha-entity-vault-update-check.timer`
- Workflow helpers:
  - `lxc/bootstrap-host-workflow.sh`
  - `lxc/staging-apply-branch.sh`
  - `lxc/configure-prod-auto-apply.sh`
  - `lxc/configure-github-main-protection.sh`

### Command interface
From repo root on the host:

```bash
./lxc/update-host.sh check   # exit 10 when updates are available
./lxc/update-host.sh status  # show mode + local/remote SHA + pending state
./lxc/update-host.sh apply   # apply update with smoke checks + rollback
./lxc/update-host.sh run     # execute configured mode (for timers)
```

### Safety behavior
- `apply` requires a clean git worktree. If dirty, update is skipped.
- High-risk diffs are flagged when changes include:
  - `requirements.txt`
  - `migrations/`
  - `lxc/ha-entity-vault.service`
- In `detect_approve`, updates are written to `${HEV_UPDATE_STATE_DIR}/pending-update.json`.
- In `auto_apply`, high-risk updates are blocked unless `HEV_UPDATE_ALLOW_HIGH_RISK_AUTO=true`.
- Before applying, the updater creates a SQLite backup and prunes older backups by retention count.
- After apply, the updater runs smoke checks:
  - `systemctl is-active ha-entity-vault.service`
  - `GET /healthz` payload includes `"status":"ok"`
  - `/entities` returns `200`
  - `/` returns `303`
  - service remains active after 10s stabilization
- If smoke checks fail after apply, updater auto-rolls back code/dependencies + DB backup and re-checks health.

### Install timer automation (host systemd)
1. Copy/update configuration:
```bash
sudo cp lxc/ha-entity-vault-update.env.example /etc/default/ha-entity-vault-updater
sudo chmod 600 /etc/default/ha-entity-vault-updater
```
2. Edit `/etc/default/ha-entity-vault-updater` and set at least:
   - `HEV_UPDATE_REPO_PATH`
   - `HEV_UPDATE_MODE`
3. Install unit files:
```bash
sudo cp lxc/ha-entity-vault-update-check.service /etc/systemd/system/
sudo cp lxc/ha-entity-vault-update-check.timer /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now ha-entity-vault-update-check.timer
```
4. Run once immediately (optional):
```bash
sudo systemctl start ha-entity-vault-update-check.service
```

Default schedule in timer template: `Sun 03:00` (local host timezone), with `Persistent=true`.

### Manual approval/apply path
For `detect_approve`, approve by running:

```bash
sudo /Users/jason/Projects/homeassistant-entity-helper/lxc/update-host.sh apply
```

All updater output is written to stdout/stderr for journald capture.

### Branch -> Staging -> Main -> Production Auto-Deploy
This workflow keeps branch validation isolated from production:

1. Run one-time Ubuntu host bootstrap:
```bash
./lxc/bootstrap-host-workflow.sh
```
This creates/validates:
- `/srv/ha-entity-vault-prod`
- `/srv/ha-entity-vault-staging`
- staging container `ha-entity-vault-staging` on `:18001`

2. Configure production updater mode + timer:
```bash
sudo /srv/ha-entity-vault-prod/lxc/configure-prod-auto-apply.sh \
  --prod-path /srv/ha-entity-vault-prod \
  --container ha-entity-vault \
  --mode auto_apply \
  --branch main \
  --allow-high-risk-auto false
```

3. Configure GitHub merge gates (repo admin):
```bash
/srv/ha-entity-vault-prod/lxc/configure-github-main-protection.sh
```
This script sets required checks (`lint`, `typecheck`, `tests`, `lxc-assets`) on `main` and disables non-squash merge methods.

4. For each feature branch, deploy branch to staging:
```bash
/srv/ha-entity-vault-staging/lxc/staging-apply-branch.sh \
  --branch codex/<feature-name> \
  --staging-path /srv/ha-entity-vault-staging \
  --staging-container ha-entity-vault-staging
```

5. Validate on staging URL (`http://<host-ip>:18001`), open PR, wait for CI, then squash merge to `main`.

6. Production applies on weekly timer (`Sun 03:00` local host time) or immediately by:
```bash
sudo systemctl start ha-entity-vault-update-check.service
```

High-risk updates (`requirements.txt`, `migrations/`, `lxc/ha-entity-vault.service`) are blocked in `auto_apply` mode unless `HEV_UPDATE_ALLOW_HIGH_RISK_AUTO=true`.

## Manual LXC Setup (if not using helper script)
1. Initialize LXD and launch container:
```bash
lxd init --auto
lxc launch ubuntu:24.04 ha-entity-vault
```
2. Add bind mounts:
```bash
lxc config device add ha-entity-vault app-src disk source=/abs/path/to/repo path=/srv/ha-entity-vault shift=true
lxc config device add ha-entity-vault app-data disk source=/abs/path/to/data path=/data shift=true
```
3. Optional host port proxy:
```bash
lxc config device add ha-entity-vault web proxy listen=tcp:0.0.0.0:18000 connect=tcp:127.0.0.1:8000
```
4. Bootstrap inside container:
```bash
lxc exec ha-entity-vault -- bash /srv/ha-entity-vault/lxc/setup-in-container.sh
```

## Local Development (No Container)
```bash
make dev-install
make run
```

Then open `http://localhost:8000`.

## Configuration
Environment variables (see `.env.example` and `lxc/ha-entity-vault.env.example`):

- `APP_NAME`: UI/application display name.
- `SESSION_SECRET`: session signing key for CSRF/session cookies.
- `HEV_DATA_DIR`: data directory used for SQLite file (default `./data`, `/data` in LXC runtime).
- `DATABASE_URL`: optional explicit SQLAlchemy URL.
- `HA_TOKEN`: optional global token override fallback.

Profile token resolution order:
1. Environment variable defined in `profile.token_env_var` (if present and set).
2. Token stored in DB for that profile.
3. `HA_TOKEN` fallback.

## Security Notes
- Home Assistant tokens can be stored in SQLite for convenience.
- SQLite token storage is plaintext at rest by default. Use encrypted host storage and strict host permissions.
- Prefer environment variable overrides (`token_env_var`) where practical.
- Tokens are never logged by the app.
- CSRF protection is enabled for state-changing form POSTs via session-bound token.
- LXC setup runs service as non-root user `haev`.
- Rate limiting is not implemented in MVP (tracked in roadmap).

## HA API Integration (MVP)
- `GET /api/config` for connection test/version discovery.
- `GET /api/states` for on-demand full entity pull.
- `WS /api/websocket` for registry enrichment (`entity/device/area/label/floor` lists).
- Auth: `Authorization: Bearer <token>`.
- Base URLs are normalized to avoid trailing slash issues.
- TLS verify can be disabled for self-signed LAN deployments (`verify_tls=false`).

## App Endpoint Reference (Manual)
- `GET /healthz` - health check.
- `GET /settings` - profile settings page.
- `POST /profiles/select` - set active profile for current session and redirect.
- `POST /profiles` - create profile.
- `POST /profiles/{profile_id}/update` - update profile.
- `POST /profiles/{profile_id}/enable` - enable profile for active use.
- `POST /profiles/{profile_id}/disable` - disable profile (hidden from switcher/actions).
- `POST /profiles/{profile_id}/delete` - delete profile + associated sync data.
- `POST /profiles/{profile_id}/test` - test Home Assistant connection.
- `POST /profiles/{profile_id}/sync` - sync entities on demand.
- `GET /entities` - entity table with filter/pagination query params.
- `GET /entities/{entity_id}` - entity detail view.
- `GET /export/json` - export filtered entities as JSON.
- `GET /export/csv` - export filtered entities as CSV.

## Quality and CI
GitHub Actions pipeline (`.github/workflows/ci.yml`) runs:
- Ruff lint
- Mypy typecheck
- Pytest
- LXC asset checks (shell syntax for scripts)

## Testing
```bash
make test
```

Includes:
- HA client unit tests with mocked httpx transport.
- API integration tests for settings + connection test + sync + browse + export flow.

## Project Structure
- `app/main.py` - FastAPI app, routes, middleware, CSRF/session, exports.
- `app/ha_client.py` - Home Assistant API client wrapper.
- `app/models.py` - SQLModel entities.
- `app/db.py` - engine/session helpers + migration startup hook.
- `app/templates/` - server-rendered HTML views.
- `app/static/` - CSS.
- `migrations/` - Alembic env + versions.
- `tests/` - unit and API tests.
- `lxc/` - LXC deployment/update scripts and systemd units.

## Roadmap / TODO

### A) LLM-Assisted Automation and Naming Suggestions
- [ ] Provider abstraction for local and remote LLMs.
  - Notes: support Ollama/LM Studio OpenAI-compatible endpoint plus remote providers with configurable model/base URL/API key/timeout.
- [ ] Safety-first proposal workflow (no auto-apply).
  - Notes: structured JSON proposals (versioned schema), diff preview, explicit user confirmation.
- [ ] Execution pipeline with dry run / apply / rollback.
  - Notes: persist proposal lifecycle state and rollback plan per change set.
- [ ] Audit log for proposals and approvals.
  - Notes: immutable audit events with actor/timestamp/result.
- [ ] RAG-style context assembly.
  - Notes: use entity snapshots first; later include automation YAML/config and dependency references.
- [ ] Constraints to protect automation-linked entities.
  - Notes: block unsafe renames unless references are updated in same proposal.
- [ ] Proposal quality gates.
  - Notes: rule engine (naming lint/reserved prefixes/uniqueness) and optional simulated trigger-condition evaluation.

### B) Naming Conventions Assistant
- [ ] Configurable naming lint profiles.
  - Notes: recommend `domain.object_location_purpose`, lowercase, underscores, no spaces.
- [ ] Duplicate and ambiguity detection.
  - Notes: flag near-collisions and missing location/area hints.
- [ ] Rename plan generator.
  - Notes: produce coordinated updates for `entity_id`, `friendly_name`, and linked references when automation/script/scene data exists.

### C) Push Updates Back to Home Assistant
- [ ] Home Assistant WebSocket API integration.
  - Notes: entity registry operations and supported automation reads/updates.
- [ ] API-first apply workflow.
  - Notes: apply through HA-supported APIs only; otherwise generate manual YAML diffs/instructions.
- [ ] Apply Changes screen.
  - Notes: queued changes, per-change toggles, confirmation gates, rollback preview.

### D) Additional Ideas
- [ ] Entity registry/device/area enrichment.
  - Notes: ingest and link `entity_registry`, `device_registry`, `area_registry` for richer topology.
- [ ] Drift detection and snapshot diffing.
  - Notes: identify new/removed entities, attribute changes, state flapping.
- [ ] Unhealthy entity diagnostics.
  - Notes: detect unavailable frequency, stale updates, and anomalous attribute patterns.
- [ ] Dependency graph view.
  - Notes: map automations/scripts/dashboards to entities as source data expands.
- [ ] Scheduled sync and webhook triggers.
  - Notes: cron-like scheduler and external webhook trigger with shared secret.
- [ ] Backup/export bundle.
  - Notes: one-click zip of snapshots, lint report, proposals, audit log for support/debug handoff.
- [ ] Multi-instance dashboard.
  - Notes: aggregate profile health and drift summary in one view.
- [ ] Read-only guest mode.
  - Notes: scoped access profile for view/export without settings mutation.
- [ ] Attribute-key search.
  - Notes: index and query by nested attribute keys/values.

## Known MVP Limitations
- No background queue; sync is user-triggered request path.
- No built-in secret encryption at rest.
- No built-in rate limiting.
- Registry enrichment is best effort and depends on the connected HA version.

## License
MIT. See `LICENSE`.
