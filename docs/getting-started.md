# Getting Started (10 Minutes)

This guide gets you from clone to first successful sync quickly.

- Full project overview: [README.md](../README.md)
- Canonical environment configuration: [docs/configuration.md](configuration.md)
- Stable/public routes: [docs/api-reference.md](api-reference.md)
- Docker deployment and day-2 operations: [docs/deployment/docker.md](deployment/docker.md)

## Prerequisites

- Git
- Node.js 20+
- npm 10+
- Python 3.12+
- Docker Engine 24+ and Docker Compose plugin v2.20+ (Docker path only)

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

## Option B: Docker Quickstart

### 1) Clone and enter the project

```bash
git clone https://github.com/jasonpagliaro/homeassistant-entity-helper.git
cd homeassistant-entity-helper
```

### 2) Confirm Docker and Compose are available

```bash
docker --version
docker compose version
docker info
```

If `docker info` fails, use the troubleshooting section in [docs/deployment/docker.md](deployment/docker.md#troubleshooting).

### 3) Create and configure `.env`

```bash
cp .env.docker.example .env
```

Set a real value for `SESSION_SECRET` in `.env`.
Keep `DATABASE_URL=` empty for default SQLite mode.
Docker publishes the app on host port `23010` by default.  
If you need legacy behavior, set `HEV_HOST_PORT=8000` in `.env`.

### 4) Start the app

```bash
docker compose up -d --build
```

### 5) Validate health

```bash
docker compose ps
docker compose logs -f app
curl -fsS http://localhost:23010/healthz
```

Open `http://localhost:23010`.

### 6) Stop when needed

```bash
docker compose down
```

## Optional: Docker Compose Postgres Overlay

```bash
docker compose -f docker-compose.yml -f docker-compose.postgres.yml up -d --build
```

Postgres mode persists data in Docker volume `hev_pg_data`.

## First Use Workflow

1. Go to `/settings`.
2. Add a Home Assistant profile (name, base URL, long-lived token), then save.
3. Click `Test Connection` for that profile.
4. Go to `/entities` and click `Sync Now`.
5. Review entities with search and filters.
6. Export via `/export/json` or `/export/csv` as needed.
7. Optional: run entity suggestions and automation suggestion workflows.

## Next Steps

- Deployment operations, backup/restore, and troubleshooting: [docs/deployment/docker.md](deployment/docker.md)
- Full configuration and precedence details: [docs/configuration.md](configuration.md)
- Stable/public API and routes: [docs/api-reference.md](api-reference.md)
