# Docker Deployment Guide

This guide covers Docker deployment and day-2 operations for HA Entity Vault.

- Default: one app container + SQLite data volume (Compose).
- Optional: one app container + Postgres container (Compose overlay).
- Optional: single container `docker run` flow for quick smoke tests.

For a quick onboarding path, see [docs/getting-started.md](../getting-started.md).

## Prerequisites
- Docker Engine 24+.
- Docker Compose plugin v2.20+.
- A copy of `.env.docker.example` saved as `.env` with real secrets.

## Single-Host Deploy (Compose Default)
This is the recommended production-like path on one host.

```bash
cp .env.docker.example .env
docker compose up -d --build
docker compose ps
docker compose logs -f app
curl -fsS http://localhost:8000/healthz
```

The app applies Alembic migrations automatically at startup.

SQLite data is persisted in Docker volume `hev_data`.

## Command Basics (Day-2 Operations)
Default Compose commands:

```bash
docker compose up -d --build   # start/update
docker compose ps              # status
docker compose logs -f app     # logs
docker compose restart app     # restart app service
docker compose down            # stop stack
```

Data-destructive cleanup (removes volumes and persisted data):

```bash
docker compose down -v
```

Compose + Postgres overlay variants:

```bash
docker compose -f docker-compose.yml -f docker-compose.postgres.yml up -d --build
docker compose -f docker-compose.yml -f docker-compose.postgres.yml ps
docker compose -f docker-compose.yml -f docker-compose.postgres.yml logs -f app
docker compose -f docker-compose.yml -f docker-compose.postgres.yml restart app
docker compose -f docker-compose.yml -f docker-compose.postgres.yml down
docker compose -f docker-compose.yml -f docker-compose.postgres.yml down -v
```

Jump links:
- [Persistence, Backup, and Restore](#persistence-backup-and-restore)
- [Troubleshooting](#troubleshooting)

## Optional: Docker Compose + Postgres Overlay

```bash
cp .env.docker.example .env
docker compose -f docker-compose.yml -f docker-compose.postgres.yml up -d --build
```

This starts both the app and a local `postgres:16-alpine` container.
The app waits for Postgres health before startup and still runs migrations automatically.

Check status:

```bash
docker compose -f docker-compose.yml -f docker-compose.postgres.yml ps
docker compose -f docker-compose.yml -f docker-compose.postgres.yml logs -f app
curl -fsS http://localhost:8000/healthz
```

Stop:

```bash
docker compose -f docker-compose.yml -f docker-compose.postgres.yml down
```

Postgres data is persisted in volume `hev_pg_data`.

## Optional: Single Container (`docker run`)
Useful for quick smoke tests and simple hosts.

```bash
docker build -t ha-entity-vault:local .

docker volume create hev_data

docker run -d \
  --name ha-entity-vault \
  --restart unless-stopped \
  --env-file .env \
  -e HEV_DATA_DIR=/data \
  -p 8000:8000 \
  -v hev_data:/data \
  ha-entity-vault:local
```

Check status:

```bash
docker ps
docker logs -f ha-entity-vault
curl -fsS http://localhost:8000/healthz
```

Stop/remove:

```bash
docker stop ha-entity-vault
docker rm ha-entity-vault
```

## Environment Variables

| Variable | Required | Default | Notes |
| --- | --- | --- | --- |
| `SESSION_SECRET` | Yes (production) | `change-me` in examples | Use a long random secret in real deployments. |
| `SESSION_HTTPS_ONLY` | No | `false` | Set to `true` when app is served behind HTTPS/TLS. |
| `HEV_DATA_DIR` | No | `./data` (host), `/data` (container defaults) | SQLite data path. |
| `DATABASE_URL` | No | empty | Leave empty for SQLite; set Postgres URL for DB container/external DB. |
| `APP_NAME` | No | `HA Entity Vault` | UI label only. |
| `HA_TOKEN` | No | empty | Global fallback HA token. |
| `HEV_LLM_ENABLED` | No | `false` | Enable LLM features only when fully configured. |
| `HEV_LLM_BASE_URL` | No | empty | OpenAI-compatible endpoint base URL. |
| `HEV_LLM_API_KEY` | No | empty | Provider API key (or profile-scoped env var references). |
| `HEV_LLM_MODEL` | No | empty | Model ID for suggestion/draft generation. |

## Persistence, Backup, and Restore

### SQLite (default profile)
Backup:

```bash
docker run --rm -v hev_data:/data -v "$PWD":/backup busybox \
  sh -c 'cp /data/ha_entity_vault.db /backup/ha_entity_vault.db.backup'
```

Restore:

```bash
docker run --rm -v hev_data:/data -v "$PWD":/backup busybox \
  sh -c 'cp /backup/ha_entity_vault.db.backup /data/ha_entity_vault.db'
```

### Postgres (overlay profile)
Backup:

```bash
docker compose -f docker-compose.yml -f docker-compose.postgres.yml exec -T postgres \
  pg_dump -U "${HEV_POSTGRES_USER:-hev}" "${HEV_POSTGRES_DB:-ha_entity_vault}" > ha_entity_vault.sql
```

Restore:

```bash
cat ha_entity_vault.sql | docker compose -f docker-compose.yml -f docker-compose.postgres.yml exec -T postgres \
  psql -U "${HEV_POSTGRES_USER:-hev}" "${HEV_POSTGRES_DB:-ha_entity_vault}"
```

## Upgrades
1. Pull latest code.
2. Review `.env` for newly added variables.
3. Run `docker compose up -d --build` (or the Postgres overlay variant).
4. Validate health with `curl -fsS http://localhost:8000/healthz`.

## Reverse Proxy and TLS Notes
- Terminate TLS at your reverse proxy (for example Nginx, Caddy, Traefik).
- Forward standard proxy headers (`X-Forwarded-For`, `X-Forwarded-Proto`).
- Set `SESSION_HTTPS_ONLY=true` when traffic is HTTPS at the edge.
- Keep `--proxy-headers` enabled (already set in container command).

## Troubleshooting

### App container is unhealthy
- Check logs: `docker compose logs -f app`.
- Verify health endpoint manually: `curl -v http://localhost:8000/healthz`.

### Migration/startup failures
- Validate `DATABASE_URL` format.
- Confirm DB service is healthy (Postgres overlay): `docker compose ... ps`.
- Check for permission issues on mounted volumes.

### SQLite permission errors
- Verify the container can write to `/data`.
- If needed, recreate volume and restart:

```bash
docker compose down -v
docker compose up -d --build
```

### Postgres connection failures
- Confirm credentials in `.env` match compose variables.
- Ensure password values are URL-safe in `DATABASE_URL` (encode special characters if needed).

## Scaling Limitation
Run one app replica by default. Suggestion processing relies on an in-process queue, so multi-replica deployments can lead to inconsistent queue behavior unless queue architecture is redesigned.
