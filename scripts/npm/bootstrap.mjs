import { spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { existsSync, readFileSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
export const repoRoot = path.resolve(scriptDir, '..', '..');
const venvDir = path.join(repoRoot, '.venv');
const requirementsFiles = ['requirements.txt', 'requirements-dev.txt'];
const requirementsStampPath = path.join(venvDir, '.requirements.sha256');

function runCommand(command, args, options = {}) {
  const result = spawnSync(command, args, {
    cwd: repoRoot,
    stdio: 'inherit',
    env: process.env,
    ...options,
  });

  if (result.error) {
    throw result.error;
  }

  if (result.status !== 0) {
    throw new Error(`Command failed: ${command} ${args.join(' ')}`);
  }
}

function getVersionTuple(rawVersion) {
  const parts = rawVersion.trim().split('.').map((part) => Number(part));
  if (parts.length < 2 || parts.some((part) => Number.isNaN(part))) {
    throw new Error(`Unable to parse Python version: '${rawVersion}'`);
  }
  return [parts[0], parts[1]];
}

function ensurePythonVersion(command) {
  const result = spawnSync(
    command,
    ['-c', 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")'],
    {
      cwd: repoRoot,
      encoding: 'utf8',
      stdio: ['ignore', 'pipe', 'pipe'],
      env: process.env,
    },
  );

  if (result.error) {
    throw new Error(`Failed to execute '${command}': ${result.error.message}`);
  }

  if (result.status !== 0) {
    const stderr = (result.stderr || '').trim();
    throw new Error(`Python version check failed for '${command}'. ${stderr}`.trim());
  }

  const [major, minor] = getVersionTuple(result.stdout || '');
  if (major < 3 || (major === 3 && minor < 12)) {
    throw new Error(`Python 3.12+ is required. Found ${major}.${minor} via '${command}'.`);
  }

  return `${major}.${minor}`;
}

function selectedPythonCommand() {
  if (process.env.PYTHON && process.env.PYTHON.trim()) {
    return process.env.PYTHON.trim();
  }
  return 'python3';
}

function venvPythonPath() {
  if (process.platform === 'win32') {
    return path.join(venvDir, 'Scripts', 'python.exe');
  }
  return path.join(venvDir, 'bin', 'python');
}

function computeRequirementsHash() {
  const hash = createHash('sha256');
  for (const file of requirementsFiles) {
    const filePath = path.join(repoRoot, file);
    if (!existsSync(filePath)) {
      throw new Error(`Missing required dependency file: ${file}`);
    }
    const content = readFileSync(filePath, 'utf8');
    hash.update(file);
    hash.update('\n');
    hash.update(content);
    hash.update('\n');
  }
  return hash.digest('hex');
}

function readRequirementsStamp() {
  if (!existsSync(requirementsStampPath)) {
    return '';
  }
  return readFileSync(requirementsStampPath, 'utf8').trim();
}

function writeRequirementsStamp(value) {
  writeFileSync(requirementsStampPath, `${value}\n`, 'utf8');
}

export function ensureBootstrap() {
  const pythonCommand = selectedPythonCommand();
  const version = ensurePythonVersion(pythonCommand);
  const venvPython = venvPythonPath();

  if (!existsSync(venvPython)) {
    console.log(`[bootstrap] Creating virtual environment at ${venvDir}`);
    runCommand(pythonCommand, ['-m', 'venv', venvDir]);
  }

  const expectedHash = computeRequirementsHash();
  const currentHash = readRequirementsStamp();

  if (expectedHash !== currentHash) {
    console.log('[bootstrap] Installing Python dependencies from requirements-dev.txt');
    runCommand(venvPython, ['-m', 'pip', 'install', '--upgrade', 'pip']);
    runCommand(venvPython, ['-m', 'pip', 'install', '-r', 'requirements-dev.txt']);
    writeRequirementsStamp(expectedHash);
  } else {
    console.log('[bootstrap] Python dependencies are up to date');
  }

  console.log(`[bootstrap] Ready with Python ${version}`);

  return {
    pythonCommand,
    venvPython,
  };
}

const entrypointPath = process.argv[1] ? path.resolve(process.argv[1]) : '';
if (entrypointPath === fileURLToPath(import.meta.url)) {
  try {
    ensureBootstrap();
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    console.error(`[bootstrap] ${message}`);
    process.exit(1);
  }
}
