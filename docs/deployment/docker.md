# Docker Deployment Guide

This guide covers standard Docker deployment flows for HA Entity Vault:

- Default: one app container + SQLite data volume.
- Optional: one app container + Postgres container.

The application runs Alembic migrations automatically during app startup.

## Prerequisites
- Docker Engine 24+.
- Docker Compose plugin v2.20+.
- A copy of `.env.docker.example` saved as `.env` with real secrets.

## Deployment Options

### Option A: Single Container (`docker run`)
This is useful for quick smoke tests and simple hosts.

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

### Option B: Docker Compose (Default SQLite Profile)
This is the recommended default path for local-first deployments.

```bash
cp .env.docker.example .env
docker compose up -d --build
```

Check status:

```bash
docker compose ps
docker compose logs -f app
curl -fsS http://localhost:8000/healthz
```

Stop:

```bash
docker compose down
```

SQLite data is persisted in volume `hev_data`.

### Option C: Docker Compose + Postgres Overlay (Optional)
Use this when you want Postgres persistence and a closer production layout.

```bash
cp .env.docker.example .env
docker compose -f docker-compose.yml -f docker-compose.postgres.yml up -d --build
```

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
3. Rebuild and restart containers:

```bash
docker compose up -d --build
```

For Postgres overlay:

```bash
docker compose -f docker-compose.yml -f docker-compose.postgres.yml up -d --build
```

Because migrations run during app startup, schema updates are applied automatically at boot.

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
