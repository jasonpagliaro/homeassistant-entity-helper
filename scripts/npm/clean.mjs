import { existsSync, rmSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(scriptDir, '..', '..');

const targets = [
  path.join(repoRoot, '.venv'),
  path.join(repoRoot, 'node_modules'),
  path.join(repoRoot, '.pytest_cache'),
  path.join(repoRoot, '.mypy_cache'),
  path.join(repoRoot, '.ruff_cache'),
];

for (const target of targets) {
  if (!existsSync(target)) {
    continue;
  }
  rmSync(target, { recursive: true, force: true });
  console.log(`[clean] removed ${path.relative(repoRoot, target)}`);
}
