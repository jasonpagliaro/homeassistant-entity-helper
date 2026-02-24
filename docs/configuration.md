# Configuration Reference

This is the canonical environment configuration reference for HA Entity Vault.

- Local template: `.env.example`
- Docker template: `.env.docker.example`

## Environment Variables

| Variable | Required | Default | Notes |
| --- | --- | --- | --- |
| `APP_NAME` | No | `HA Entity Vault` | UI/application display name. |
| `SESSION_SECRET` | Yes (production) | `dev-only-change-me` in code | Session signing key for CSRF/session cookies. Set a strong random value. |
| `SESSION_HTTPS_ONLY` | No | `false` | Set `true` behind HTTPS/TLS so session cookies are marked secure. |
| `LOG_LEVEL` | No | `INFO` | Root logger level. Typical values: `DEBUG`, `INFO`, `WARNING`, `ERROR`. |
| `HEV_DATA_DIR` | No | `./data` | Base directory for default SQLite path resolution. Compose sets `/data` in containers. |
| `DATABASE_URL` | No | empty | Explicit SQLAlchemy URL. Highest-precedence database setting. |
| `HEV_DB_PATH` | No | empty | SQLite file override path, used only when `DATABASE_URL` is unset. |
| `HA_TOKEN` | No | empty | Global fallback Home Assistant token. |
| `HEV_LLM_ENABLED` | No | `false` | Enables LLM-assisted suggestion/draft features when provider settings are valid. |
| `HEV_LLM_BASE_URL` | No | empty | OpenAI-compatible provider base URL. |
| `HEV_LLM_API_KEY` | No | empty | API key for the global `HEV_LLM_*` provider config. |
| `HEV_LLM_MODEL` | No | `gpt-4o-mini` (when unset) | Model ID for LLM requests. |
| `HEV_LLM_TIMEOUT_SECONDS` | No | `20` | Outbound LLM request timeout. |
| `HEV_LLM_MAX_CONCURRENCY` | No | `4` | Max in-flight LLM requests per run. |
| `HEV_LLM_TEMPERATURE` | No | `0.2` | LLM sampling temperature. |
| `HEV_LLM_MAX_OUTPUT_TOKENS` | No | `900` | Max generated tokens per LLM response. |
| `HEV_LLM_EXTRA_HEADERS_JSON` | No | empty | Optional JSON object of extra outbound LLM HTTP headers. |
| `HEV_AUTOMATION_DRAFT_MAX_ITEMS` | No | `50` | Max entities considered in one automation draft run. |
| `HEV_BUILD_COMMIT_SHA` | No | empty | Build SHA override for update-checker version detection when `.git` metadata is unavailable. |

## Provider API Key Environment Variables

Profile-scoped LLM connections store env var names (not plaintext secrets).  
Common examples:

- `OPENAI_API_KEY`
- `OPENROUTER_API_KEY`
- `LM_STUDIO_API_KEY`
- Any custom variable name referenced by `llm_connections.api_key_env_var`

## Resolution and Precedence Rules

### Database URL Resolution

The app resolves database settings in this order:

1. `DATABASE_URL` (if set)
2. `HEV_DB_PATH` (if set, converted to `sqlite:///...`)
3. SQLite default at `HEV_DATA_DIR/ha_entity_vault.db`

### Home Assistant Token Resolution

For each profile, the app resolves token value in this order:

1. Environment variable named by `profile.token_env_var` (if configured and set)
2. Token stored in DB for that profile
3. `HA_TOKEN` fallback

### LLM API Key Resolution

For profile-scoped automation suggestions:

1. Environment variable named by `llm_connections.api_key_env_var` (if configured and set)
2. No DB plaintext API key fallback (run fails if key is required and missing)

For global LLM settings, `HEV_LLM_API_KEY` is used with `HEV_LLM_BASE_URL`/`HEV_LLM_MODEL`.

## Deployment Notes

- For Docker Compose default (SQLite), keep `DATABASE_URL=` empty and use persistent volume `hev_data`.
- For Compose Postgres overlay, use `docker-compose.postgres.yml` and `HEV_POSTGRES_*` variables from `.env.docker.example`.
- For deterministic version tracking in container builds, pass `HEV_BUILD_COMMIT_SHA` at build or runtime.
