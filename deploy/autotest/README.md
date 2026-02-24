# Auto-Test Runner Assets

- `runner.Dockerfile`: Linux runner image with pinned Playwright + Chromium preinstalled.
- `package.json`: pinned Playwright dependency and helper commands.
- `scripts/playwright-preflight.mjs`: mandatory browser launch preflight (fail-fast).
- `scripts/assert-config-timestamps.mjs`: `/config` timestamp check with explicit browser required/skip modes.

See `docs/deployment/auto-test-runner.md` for setup and usage.
