# Docker Deployment Guide

This guide covers Docker deployment and day-2 operations for HA Entity Vault.

- Default: one app container + SQLite data volume (Compose).
- Optional: one app container + Postgres container (Compose overlay).
- Optional: single container `docker run` flow for quick smoke tests.

For a quick onboarding path, see [docs/getting-started.md](../getting-started.md).  
For canonical environment definitions and precedence rules, see [docs/configuration.md](../configuration.md).

## Prerequisites
- Docker Engine 24+.
- Docker Compose plugin v2.20+.
- A copy of `.env.docker.example` saved as `.env` with real secrets.

## Preflight Checks
Run these checks from the repository root before deployment or troubleshooting:

```bash
pwd
ls docker-compose.yml docker-compose.postgres.yml .env.docker.example
docker --version
docker compose version
docker info
```

Expected results:
- The compose and env template files are found in the current directory.
- `docker --version` and `docker compose version` both return version output.
- `docker info` returns Server details (not daemon connection errors).

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

Use [docs/configuration.md](../configuration.md) as the canonical environment reference.
The table below only lists Docker deployment specifics.

| Variable | Required | Default | Notes |
| --- | --- | --- | --- |
| `SESSION_SECRET` | Yes (production) | placeholder in `.env.docker.example` | Set to a long random value before first deployment. |
| `HEV_DATA_DIR` | No | `/data` in Compose app container | Container path for SQLite persistence volume mapping. |
| `DATABASE_URL` | No | empty | Leave empty for SQLite Compose default. Set for external DB or advanced overrides. |
| `HEV_POSTGRES_DB` | No | `ha_entity_vault` | Database name used by the Postgres overlay service. |
| `HEV_POSTGRES_USER` | No | `hev` | Username used by the Postgres overlay service. |
| `HEV_POSTGRES_PASSWORD` | No | `hev_change_me` | Password used by the Postgres overlay service. Change before production use. |
| `HEV_BUILD_COMMIT_SHA` | No | empty | Optional local build SHA used by `/config` update checker when `.git` metadata is unavailable in runtime containers. |

To provide deterministic update-checker local version info in containers:

```bash
docker build \
  --build-arg HEV_BUILD_COMMIT_SHA="$(git rev-parse HEAD)" \
  -t ha-entity-vault:local .
```

Or set `HEV_BUILD_COMMIT_SHA` in `.env` for Compose/runtime injection.

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

### Common Errors (Top 8)

| Symptom | Likely Cause | Fix |
| --- | --- | --- |
| `docker: command not found` | Docker CLI is not installed or shell session is stale. | Install Docker Desktop (macOS/Windows) or Docker Engine + Compose plugin (Linux), then re-run version checks. |
| `Cannot connect to the Docker daemon` | Docker daemon is not running. | Start Docker Desktop (macOS/Windows) or start the `docker` service (Linux). |
| `permission denied /var/run/docker.sock` | Current Linux user is not in the `docker` group. | Add user to `docker` group, then re-login or refresh group membership. |
| `docker compose` not recognized | Compose plugin is missing or old command syntax is used. | Install/update Compose plugin and use `docker compose` (with a space). |
| `.env` missing or placeholder `SESSION_SECRET` | Environment file was not created or not updated. | Copy `.env.docker.example` to `.env`, set a real `SESSION_SECRET`, keep `DATABASE_URL=` for SQLite mode. |
| Port `8000` already in use | Another process or container is bound to `8000`. | Stop conflicting process/container or map to another host port. |
| App container unhealthy/startup crash | Runtime, migration, or env config issue during startup. | Inspect app logs and restart with a clean rebuild. |
| Postgres overlay auth/connect failures | `HEV_POSTGRES_*` vars mismatch or stale Postgres volume state. | Verify `.env` values, restart overlay, and reset Postgres volume if credentials changed. |

### 1) `docker: command not found`
- **What you'll see:** `zsh: command not found: docker` or similar.
- **Run this check:**

```bash
docker --version
```

- **Fix:**

```bash
# Ubuntu/Debian example
sudo apt-get update
sudo apt-get install -y docker.io docker-compose-plugin
```

On macOS or Windows, install/start Docker Desktop, then open a new terminal.

- **Re-test:**

```bash
docker --version
docker compose version
docker compose up -d --build
curl -fsS http://localhost:8000/healthz
```

### 2) `Cannot connect to the Docker daemon`
- **What you'll see:** `Cannot connect to the Docker daemon at unix:///var/run/docker.sock`.
- **Run this check:**

```bash
docker info
```

- **Fix:**

```bash
# Linux
sudo systemctl start docker
sudo systemctl enable docker
```

On macOS or Windows, start Docker Desktop and wait until it reports "Engine running."

- **Re-test:**

```bash
docker info
docker compose up -d --build
curl -fsS http://localhost:8000/healthz
```

### 3) `permission denied /var/run/docker.sock` (Linux)
- **What you'll see:** `permission denied while trying to connect to the Docker daemon socket`.
- **Run this check:**

```bash
id -nG
```

- **Fix:**

```bash
sudo usermod -aG docker "$USER"
newgrp docker
```

If `newgrp` is not available, log out and back in.

- **Re-test:**

```bash
docker info
docker compose up -d --build
curl -fsS http://localhost:8000/healthz
```

### 4) `docker compose` not recognized / old Compose
- **What you'll see:** `docker: 'compose' is not a docker command` or only `docker-compose` works.
- **Run this check:**

```bash
docker compose version
```

- **Fix:**

```bash
# Ubuntu/Debian example
sudo apt-get update
sudo apt-get install -y docker-compose-plugin
```

Use `docker compose` (space), not `docker-compose` (hyphen), in this repo.

- **Re-test:**

```bash
docker compose version
docker compose up -d --build
curl -fsS http://localhost:8000/healthz
```

### 5) `.env` missing or placeholder `SESSION_SECRET` not replaced
- **What you'll see:** startup/config errors, or insecure placeholder secret left in `.env`.
- **Run this check:**

```bash
ls -l .env && grep '^SESSION_SECRET=' .env && grep '^DATABASE_URL=' .env
```

- **Fix:**

```bash
cp .env.docker.example .env
openssl rand -hex 32
```

Set the generated value as `SESSION_SECRET=` in `.env`, and keep `DATABASE_URL=` empty for SQLite mode.

- **Re-test:**

```bash
docker compose up -d --build
curl -fsS http://localhost:8000/healthz
```

### 6) Port `8000` already in use
- **What you'll see:** `Bind for 0.0.0.0:8000 failed: port is already allocated`.
- **Run this check:**

```bash
lsof -iTCP:8000 -sTCP:LISTEN
```

- **Fix:**

```bash
docker compose down
docker ps --format '{{.Names}}\t{{.Ports}}'
```

Stop the process/container already using port `8000`, or change the app port mapping in `docker-compose.yml` (for example `8001:8000`).

- **Re-test:**

```bash
docker compose up -d --build
curl -fsS http://localhost:8000/healthz
```

### 7) App container unhealthy or startup crash
- **What you'll see:** `docker compose ps` shows `unhealthy`, `restarting`, or exits quickly.
- **Run this check:**

```bash
docker compose logs --tail=200 app
```

- **Fix:**

```bash
docker compose down
docker compose up -d --build
docker compose logs --tail=200 app
```

If logs show DB/migration errors, verify `DATABASE_URL` and Postgres overlay settings.

- **Re-test:**

```bash
docker compose ps
curl -fsS http://localhost:8000/healthz
```

### 8) Postgres overlay connection/auth failures
- **What you'll see:** app logs include authentication failures or cannot connect to `postgres:5432`.
- **Run this check:**

```bash
docker compose -f docker-compose.yml -f docker-compose.postgres.yml logs --tail=200 postgres app
```

- **Fix:**

```bash
docker compose -f docker-compose.yml -f docker-compose.postgres.yml down
docker compose -f docker-compose.yml -f docker-compose.postgres.yml up -d --build
```

If credentials were changed after first startup, reset Postgres state (data-destructive):

```bash
docker compose -f docker-compose.yml -f docker-compose.postgres.yml down -v
docker compose -f docker-compose.yml -f docker-compose.postgres.yml up -d --build
```

- **Re-test:**

```bash
docker compose -f docker-compose.yml -f docker-compose.postgres.yml ps
curl -fsS http://localhost:8000/healthz
```

## Scaling Limitation
Run one app replica by default. Suggestion processing relies on an in-process queue, so multi-replica deployments can lead to inconsistent queue behavior unless queue architecture is redesigned.
