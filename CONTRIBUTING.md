# Contributing

## Development setup
1. Install Node.js 20+ and npm 10+.
2. Install Python 3.12+.
3. Install dependencies: `npm ci`.
4. Start app locally: `npm run run`.

If your default `python3` is not 3.12+, set `PYTHON`, for example:

```bash
PYTHON=python3.12 npm run build
```

## Quality checks
- Full gate: `npm run build`
- Docs checks: `npm run docs:check`
- Lint: `npm run lint`
- Typecheck: `npm run typecheck`
- Tests: `npm run test`
- Format: `npm run format`

## Documentation updates
- Canonical environment documentation lives in `docs/configuration.md`.
- Stable/public route documentation lives in `docs/api-reference.md`.
- Keep cross-links current when moving or renaming docs pages.
- Run `npm run docs:check` before opening a PR that changes markdown content.

## Compatibility make targets
`make` remains available for compatibility and delegates to npm scripts:
- `make build`
- `make lint`
- `make typecheck`
- `make test`
- `make run`

## Pull requests
1. Keep PRs focused and include tests for behavior changes.
2. Do not commit real Home Assistant tokens or private env files.
3. Update docs (`README.md` and `docs/`) when adding or changing features, routes, or operational workflows.
4. Ensure CI (`build`) is green before requesting review.

## Code style
- Keep modules small and testable.
- Prefer explicit data flow over implicit globals.
- Avoid logging secrets (especially tokens).
