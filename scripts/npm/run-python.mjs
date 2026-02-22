import { spawnSync } from 'node:child_process';

import { ensureBootstrap, repoRoot } from './bootstrap.mjs';

const args = process.argv.slice(2);
if (args.length === 0) {
  console.error('Usage: node scripts/npm/run-python.mjs <python args>');
  process.exit(64);
}

let venvPython;
try {
  ({ venvPython } = ensureBootstrap());
} catch (error) {
  const message = error instanceof Error ? error.message : String(error);
  console.error(`[run-python] ${message}`);
  process.exit(1);
}

const result = spawnSync(venvPython, args, {
  cwd: repoRoot,
  stdio: 'inherit',
  env: process.env,
});

if (result.error) {
  console.error(`[run-python] ${result.error.message}`);
  process.exit(1);
}

process.exit(result.status ?? 1);
