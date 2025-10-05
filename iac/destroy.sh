#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

if [[ ! -f package-lock.json && ! -d node_modules ]]; then
  npm install
fi

npx cdk destroy --force "$@"
