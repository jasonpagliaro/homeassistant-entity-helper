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
- Lint: `npm run lint`
- Typecheck: `npm run typecheck`
- Tests: `npm run test`
- Format: `npm run format`

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
3. Update docs (`README.md`) when adding features or endpoints.
4. Ensure CI (`build`) is green before requesting review.

## Code style
- Keep modules small and testable.
- Prefer explicit data flow over implicit globals.
- Avoid logging secrets (especially tokens).
