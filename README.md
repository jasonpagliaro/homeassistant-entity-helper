# HA Entity Vault

HA Entity Vault is a self-hosted standalone app for pulling Home Assistant entities on demand, browsing/filtering them, and exporting filtered views as JSON/CSV.

- Stack: Python 3.12, FastAPI, SQLModel + SQLite, Jinja2, httpx, Alembic.
- Build/task runner: npm (Node 20+).
- Runtime: Python/FastAPI.
- Data persistence: local `./data` by default (configurable via `HEV_DATA_DIR`).
- UI name is configurable via `APP_NAME`, so repo/product naming is easy to change later.

## Install and Run
For a concise onboarding path, see [docs/getting-started.md](docs/getting-started.md).

### Path A: Local Runtime (npm + Python)
Prerequisites:
- Node.js 20+
- npm 10+
- Python 3.12+

From repo root:

```bash
npm ci
npm run run
curl -fsS http://localhost:8000/healthz
```

Then open `http://localhost:8000`.

If your default `python3` is not 3.12+, set `PYTHON` explicitly:

```bash
PYTHON=python3.12 npm run run
```

### Path B: Docker Compose (Recommended Deployment Path)
First-time Docker user? Follow [docs/getting-started.md](docs/getting-started.md#docker-quickstart-from-zero) for clone-to-running instructions.

For deployment operations and troubleshooting, use [docs/deployment/docker.md](docs/deployment/docker.md).

Prerequisites:
- Docker 24+
- Docker Compose plugin v2.20+

From repo root:

```bash
cp .env.docker.example .env
# update SESSION_SECRET in .env before first deployment
docker compose up -d --build
curl -fsS http://localhost:8000/healthz
```

Then open `http://localhost:8000`.

SQLite data persists in the named Docker volume `hev_data`.

Optional Postgres overlay (instead of SQLite):

```bash
docker compose -f docker-compose.yml -f docker-compose.postgres.yml up -d --build
```

Postgres overlay data persists in the named Docker volume `hev_pg_data`.

For deployment operations (backup/restore, reverse proxy, troubleshooting), see [docs/deployment/docker.md](docs/deployment/docker.md).

## First Use Workflow (Core)
1. Open `/settings`, add a profile (name, base URL, token), and save.
2. Click `Test Connection` for that profile.
3. Open `/entities`, then click `Sync Now`.
4. Apply filters/search, inspect entity details, and export via JSON/CSV.
5. Optional: run `Run Suggestions Check` and automation suggestions workflows.

## Deploy Command Basics
For day-2 Docker Compose operations:

```bash
docker compose ps              # status
docker compose logs -f app     # app logs
docker compose restart app     # restart app service
docker compose down            # stop stack
```

Destructive cleanup (removes persisted volumes/data):

```bash
docker compose down -v
```

For full Docker command variants (including Postgres overlay), see [docs/deployment/docker.md](docs/deployment/docker.md).

## What's New
- 2026-02-22: Header/page navigation was cleaned up with accessible primary tabs, active-route highlighting, mobile horizontal scrolling, and API Docs moved to a global footer utility link. See `CHANGELOG.md` for details.

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

## Build and Quality Commands
```bash
npm run bootstrap      # create/update .venv and install requirements-dev.txt
npm run build          # full quality gate: lint + typecheck + tests
npm run lint
npm run typecheck
npm run test
npm run format
npm run db:migrate
npm run run
npm run clean
```

## Make Compatibility Targets
`make` targets are still available and delegate to npm scripts:

```bash
make install
make dev-install
make build
make lint
make typecheck
make test
make format
make run
make alembic-upgrade
make clean
```

## Configuration
Environment variables (see `.env.example`):

- `APP_NAME`: UI/application display name.
- `SESSION_SECRET`: session signing key for CSRF/session cookies.
- `SESSION_HTTPS_ONLY`: set `true` behind HTTPS/TLS terminators to mark session cookies as secure.
- `HEV_DATA_DIR`: data directory used for SQLite file (default `./data`).
- `DATABASE_URL`: optional explicit SQLAlchemy URL.
- `HA_TOKEN`: optional global token override fallback.
- `HEV_LLM_ENABLED`: enable LLM-assisted suggestions/drafts when true and fully configured.
- `HEV_LLM_BASE_URL`: OpenAI-compatible provider base URL.
- `HEV_LLM_API_KEY`: provider API key.
- `HEV_LLM_MODEL`: model ID for suggestion/draft generation.
- `HEV_LLM_TIMEOUT_SECONDS`: outbound LLM request timeout.
- `HEV_LLM_MAX_CONCURRENCY`: max in-flight LLM requests per run.
- `HEV_AUTOMATION_DRAFT_MAX_ITEMS`: cap candidates processed in one draft generation run.
- `HEV_BUILD_COMMIT_SHA`: optional build commit SHA override used by the update checker when `.git` metadata is unavailable.
- `OPENAI_API_KEY` / `OPENROUTER_API_KEY` / custom vars: examples of env vars referenced by profile-scoped LLM connections.

Profile token resolution order:
1. Environment variable defined in `profile.token_env_var` (if present and set).
2. Token stored in DB for that profile.
3. `HA_TOKEN` fallback.

LLM API key resolution for profile-scoped automation suggestions:
1. Environment variable defined in `llm_connections.api_key_env_var` (if present and set).
2. No plaintext API key fallback in DB (suggestion run fails if key is required and missing).

## Update Checker
- App-level update settings and status live on `GET /config`.
- Source of truth is GitHub commit head for configured `{owner}/{repo}@{branch}`.
- Auto-check runs only when `/config` loads and the configured interval has elapsed.
- Manual checks run via `POST /config/check-updates`.
- Update banner dismissal is deployment-scoped (DB-backed) and reappears automatically when a newer commit is detected.
- If local SHA cannot be resolved, status becomes `unknown_local_sha`. Set `HEV_BUILD_COMMIT_SHA` for deterministic deployment version tracking.

## Security Notes
- Home Assistant tokens can be stored in SQLite for convenience.
- SQLite token storage is plaintext at rest by default. Use encrypted host storage and strict host permissions.
- Prefer environment variable overrides (`token_env_var`) where practical.
- Tokens are never logged by the app.
- CSRF protection is enabled for state-changing form POSTs via session-bound token.
- Rate limiting is not implemented in MVP (tracked in roadmap).

## HA API Integration (MVP)
- `GET /api/config` for connection test/version discovery.
- `GET /api/states` for on-demand full entity pull.
- `WS /api/websocket` for registry enrichment (`entity/device/area/label/floor` lists).
- Auth: `Authorization: Bearer <token>`.
- Base URLs are normalized to avoid trailing slash issues.
- TLS verify can be disabled for self-signed LAN deployments (`verify_tls=false`).

## Entity Suggestions Policy
- Candidate scope is intentionally limited to actionable domains: `sensor`, `binary_sensor`, `lock`.
- `event` entities are excluded from suggestion runs.
- Missing area is a blocker only when at least one candidate in the run has area/device enrichment.
- If a run has no area/device enrichment at all, missing area is downgraded to review warning for that run.
- If enrichment is unavailable, verify Home Assistant WebSocket/registry access and rerun suggestions after connectivity or permission fixes.
- Existing historical suggestion runs are not backfilled when policy changes; run a new suggestions check to apply current scoring logic.
- Missing details workflow supports direct HA registry updates for area, friendly name, sensor/binary-sensor `device_class`, and labels.
- Workflow writes require an HA admin-capable token because registry update commands are admin-only.
- Use workflow queue for batched edits, then run one batch recheck (`sync + suggestions`) to verify updated statuses.
- Manual-only issues remain visible with guidance and are not auto-editable in the workflow.

## App Endpoint Reference (Manual)
- `GET /healthz` - health check.
- `GET /config` - app-level update checker status and settings page.
- `POST /config/update-settings` - persist update checker settings.
- `POST /config/check-updates` - run update check now.
- `POST /config/update-banner/dismiss` - dismiss current update banner globally.
- `POST /config/update-banner/reset` - clear banner dismissal state.
- `GET /settings` - profile settings page.
- `POST /profiles/select` - set active profile for current session and redirect.
- `POST /profiles` - create profile.
- `POST /profiles/{profile_id}/update` - update profile.
- `POST /profiles/{profile_id}/enable` - enable profile for active use.
- `POST /profiles/{profile_id}/disable` - disable profile (hidden from switcher/actions).
- `POST /profiles/{profile_id}/delete` - delete profile + associated sync data.
- `POST /profiles/{profile_id}/test` - test Home Assistant connection.
- `POST /profiles/{profile_id}/sync` - sync entities on demand.
- `POST /profiles/{profile_id}/run-entity-suggestions` - run readiness suggestion checks.
- `POST /profiles/{profile_id}/generate-automation-drafts` - generate automation drafts from suggestions.
- `POST /profiles/{profile_id}/llm-connections` - create profile-scoped LLM connection.
- `POST /llm-connections/{id}/update` - update LLM connection.
- `POST /llm-connections/{id}/test` - test LLM connection.
- `POST /llm-connections/{id}/delete` - delete LLM connection.
- `POST /profiles/{profile_id}/suggestions/runs` - queue automation suggestion run (suggest-only; no auto-apply).
- `GET /entities` - entity table with filter/pagination query params.
- `GET /entities/{entity_id}` - entity detail view.
- `GET /suggestions` - automation suggestion run list.
- `GET /suggestions/{run_id}` - automation suggestion run detail and proposal review.
- `POST /suggestions/proposals/{proposal_id}/status` - mark proposal accepted/rejected.
- `GET /entity-suggestions` - readiness suggestion list.
- `GET /entity-suggestions/{suggestion_id}` - readiness suggestion detail.
- `GET /entity-suggestions/workflow` - missing details workflow queue.
- `GET /entity-suggestions/{suggestion_id}/workflow` - missing details workflow detail/action view.
- `POST /entity-suggestions/{suggestion_id}/workflow/apply` - apply workflow changes directly to HA registry.
- `POST /entity-suggestions/{suggestion_id}/workflow/skip` - skip workflow item for now.
- `POST /profiles/{profile_id}/entity-suggestions/recheck` - run batch verification (`sync` then `suggestions`).
- `GET /automation-drafts` - automation draft list.
- `GET /automation-drafts/{draft_id}` - automation draft detail.
- `POST /automation-drafts/{draft_id}/accept` - mark draft accepted.
- `POST /automation-drafts/{draft_id}/reject` - mark draft rejected.
- `GET /export/json` - export filtered entities as JSON.
- `GET /export/csv` - export filtered entities as CSV.
- `GET /api/entity-suggestions` - suggestions API (list + filter/pagination).
- `GET /api/entity-suggestions/{suggestion_id}` - suggestion API detail.
- `GET /api/suggestions/runs/{run_id}` - automation suggestion run status API.
- `GET /api/automation-drafts` - draft API (list + filter/pagination).
- `GET /api/automation-drafts/{draft_id}` - draft API detail.

## Quality and CI
GitHub Actions pipeline (`.github/workflows/ci.yml`) runs one authoritative quality gate:
- `npm run build` (lint, typecheck, tests)

Optional helper for GitHub main branch protection:
- `scripts/github/configure-main-protection.sh`
- Configures required check: `build` (plus squash-only merge policy)

## Testing
```bash
npm run test
```

## Project Structure
- `app/main.py` - FastAPI app, routes, middleware, CSRF/session, exports.
- `app/ha_client.py` - Home Assistant API client wrapper.
- `app/models.py` - SQLModel entities.
- `app/db.py` - engine/session helpers + migration startup hook.
- `Dockerfile` - production-oriented image build definition.
- `docker-compose.yml` - default Docker deployment (SQLite).
- `docker-compose.postgres.yml` - optional Postgres overlay for Docker deployments.
- `docs/getting-started.md` - 10-minute install, run, and first-use guide.
- `docs/deployment/docker.md` - Docker operations and deployment guide.
- `app/templates/` - server-rendered HTML views.
- `app/static/` - CSS.
- `migrations/` - Alembic env + versions.
- `tests/` - unit and API tests.
- `scripts/npm/` - npm-to-Python bootstrap and execution helpers.
- `scripts/github/` - repository administration helpers.

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
- Suggestion runs use an in-process worker queue; run one app replica unless queue orchestration is redesigned.

## License
MIT. See `LICENSE`.
