# Getting Started (10 Minutes)

This guide gets you from clone to first successful sync quickly.

For complete project details, see [README.md](../README.md).  
For Docker operations (backup/restore, reverse proxy, troubleshooting), see [docs/deployment/docker.md](deployment/docker.md).

## Prerequisites
- Node.js 20+
- npm 10+
- Python 3.12+
- Docker Engine 24+ and Docker Compose plugin v2.20+ (for container deployment)

## Option A: Local Run
From repo root:

```bash
npm ci
npm run run
curl -fsS http://localhost:8000/healthz
```

If your default Python is not 3.12+, run:

```bash
PYTHON=python3.12 npm run run
```

Open `http://localhost:8000`.

## Option B: Docker Compose Deploy (Default)
From repo root:

```bash
cp .env.docker.example .env
docker compose up -d --build
docker compose ps
docker compose logs -f app
curl -fsS http://localhost:8000/healthz
```

Open `http://localhost:8000`.

Optional Postgres overlay:

```bash
docker compose -f docker-compose.yml -f docker-compose.postgres.yml up -d --build
```

## First Use Workflow (Core)
1. Go to `/settings`.
2. Add a Home Assistant profile (name, base URL, long-lived token), then save.
3. Click `Test Connection` on that profile.
4. Go to `/entities` and click `Sync Now`.
5. Use search/filters to inspect entities.
6. Export filtered results via `/export/json` or `/export/csv`.
7. Optional: run entity suggestions and automation suggestions flows.

## Data Storage
- Local run defaults to SQLite at `./data/ha_entity_vault.db` (unless `DATABASE_URL` is set).
- Docker Compose default persists SQLite data in Docker volume `hev_data`.
- Docker Compose Postgres overlay persists Postgres data in Docker volume `hev_pg_data`.

## Next Steps
- Day-2 Docker commands, backups, restore, and troubleshooting: [docs/deployment/docker.md](deployment/docker.md)
- Full app features and endpoint reference: [README.md](../README.md)
