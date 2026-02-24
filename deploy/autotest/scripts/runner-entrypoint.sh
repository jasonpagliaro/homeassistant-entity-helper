#!/usr/bin/env sh
set -eu

node scripts/playwright-preflight.mjs

if [ "$#" -gt 0 ]; then
  exec "$@"
fi
