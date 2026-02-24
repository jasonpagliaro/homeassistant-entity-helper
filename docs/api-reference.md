# API Reference

This page documents stable/public routes for HA Entity Vault.

- Full generated API schema: `GET /docs` (FastAPI OpenAPI UI).
- This page is intentionally not exhaustive.
- Internal form-action and workflow mutation routes are implementation details and may change without notice.

## Health and Metadata

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/healthz` | Liveness and readiness health check used by deployments and probes. |
| `GET` | `/version` | Resolved app commit metadata for deployed build tracking. |
| `GET` | `/update-status` | Runtime auto-update state and DB-backed update/check status summary. |

## User-Facing HTML Routes

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/` | App entry route. |
| `GET` | `/settings` | Home Assistant profile and LLM connection management UI. |
| `GET` | `/config` | App-level settings and update checker page. |
| `GET` | `/entities` | Entity list with search/filter/pagination UI. |
| `GET` | `/entities/{entity_id}` | Entity detail page. |
| `GET` | `/config-items` | Configuration item list page. |
| `GET` | `/config-items/{snapshot_id}` | Configuration item detail page. |
| `GET` | `/entity-suggestions` | Entity readiness suggestions page. |
| `GET` | `/entity-suggestions/{suggestion_id}` | Entity suggestion detail page. |
| `GET` | `/entity-suggestions/workflow` | Missing-details workflow queue page. |
| `GET` | `/entity-suggestions/{suggestion_id}/workflow` | Missing-details workflow detail page. |
| `GET` | `/suggestions` | Automation suggestion runs page. |
| `GET` | `/suggestions/{run_id}` | Automation suggestion run detail page. |
| `GET` | `/suggestions/queue` | Proposal queue and review page. |
| `GET` | `/automation-drafts` | Automation draft list page. |
| `GET` | `/automation-drafts/{draft_id}` | Automation draft detail page. |

## API Routes

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/api/entity-suggestions` | Entity suggestions API list with filter/pagination. |
| `GET` | `/api/entity-suggestions/{suggestion_id}` | Entity suggestion API detail. |
| `GET` | `/api/suggestions/runs/{run_id}` | Automation suggestion run status API. |
| `GET` | `/api/automation-drafts` | Automation draft API list with filter/pagination. |
| `GET` | `/api/automation-drafts/{draft_id}` | Automation draft API detail. |
| `GET` | `/api/llm/presets` | LLM provider preset metadata for UI forms. |
| `POST` | `/api/llm/models` | Fetch available models from a configured LLM provider. |
| `POST` | `/api/llm/test-draft` | Validate prompt/provider behavior with a draft test call. |

## Export Routes

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/export/json` | Export currently filtered entities as JSON. |
| `GET` | `/export/csv` | Export currently filtered entities as CSV. |
