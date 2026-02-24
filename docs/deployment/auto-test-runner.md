# Auto-Test Browser Runner (Linux)

This guide defines the browser runtime contract for commit-level UI checks that render `GET /config`.

## Why this exists

The auto-test harness must not install Playwright/Chromium ad hoc during test execution.

- Browser runtime is preinstalled at runner image build time.
- Browser startup is validated by a mandatory preflight step.
- Runtime install fallback is intentionally disabled.

## Pinned versions and cache

- Node: 20+
- Playwright: pinned in `deploy/autotest/package.json` (`1.52.0`)
- Browser cache path: `PLAYWRIGHT_BROWSERS_PATH=/ms-playwright`

## Build the runner image

From repository root:

```bash
docker build \
  -f deploy/autotest/runner.Dockerfile \
  -t hev-autotest-runner:pw-1.52.0 \
  --build-arg PLAYWRIGHT_VERSION=1.52.0 \
  .
```

The Dockerfile performs these build-time steps:

1. Installs pinned npm dependency (`playwright`).
2. Installs Chromium with:
   - `npx -y playwright@<pinned> install --with-deps chromium`
3. Runs preflight (`scripts/playwright-preflight.mjs`) and fails the image build if launch fails.

## Mandatory preflight at test startup

The runner entrypoint always runs preflight first. If you pass a command, it executes that command after preflight succeeds.

Preflight-only:

```bash
docker run --rm hev-autotest-runner:pw-1.52.0
```

Expected output:

```text
[preflight] Chromium launch succeeded. Browser runtime is ready.
```

If runtime is missing/corrupted, preflight exits non-zero with an actionable error.

## Timestamp check contract for `/config`

Use `assert-config-timestamps.mjs` for commit-level checks.

### Rendered check (default)

- Requires browser runtime.
- Verifies each `time[data-local-datetime][datetime]` node is rendered to localized visible text.

```bash
docker run --rm \
  --network host \
  -e CONFIG_URL=http://127.0.0.1:8000/config \
  hev-autotest-runner:pw-1.52.0 \
  node scripts/assert-config-timestamps.mjs
```

### Intentional browser-skip fallback

When browser rendering is intentionally skipped, fallback validates HTML markers only:

```bash
docker run --rm \
  --network host \
  -e CONFIG_URL=http://127.0.0.1:8000/config \
  -e BROWSER_CHECK_MODE=skip \
  hev-autotest-runner:pw-1.52.0 \
  node scripts/assert-config-timestamps.mjs
```

Fallback checks for `time[datetime][data-local-datetime]` markers in server-rendered HTML.

## Local wrapper command

A repository wrapper is provided for harness invocation:

```bash
npm install --prefix deploy/autotest
CONFIG_URL=http://127.0.0.1:8000/config scripts/harness/run-config-timestamp-check.sh
```

By default, this performs the rendered browser check (`BROWSER_CHECK_MODE=required`).

## Removing ad-hoc install fallback

If the harness currently runs commands like runtime `npx playwright install ...` in the middle of tests, remove that path.

Replace it with:

1. Runner image provisioning at build time.
2. Mandatory preflight at test startup.
3. `assert-config-timestamps.mjs` execution in required or explicit skip mode.
