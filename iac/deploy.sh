#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

if [[ ! -f package-lock.json && ! -d node_modules ]]; then
  npm install
fi

KEY_NAME=${INSTANCE_KEY_NAME:-${EC2_KEY_NAME:-}}
if [[ -z "$KEY_NAME" ]]; then
  echo "Error: Provide an EC2 key pair name via INSTANCE_KEY_NAME env var or by editing deploy.sh." >&2
  exit 1
fi

REGION=${CDK_DEPLOY_REGION:-${AWS_REGION:-us-east-1}}

EXTRA_ARGS=()
if [[ -n "${CDK_ADDITIONAL_CONTEXT:-}" ]]; then
  EXTRA_ARGS+=($CDK_ADDITIONAL_CONTEXT)
fi

npx cdk deploy \
  --require-approval never \
  --region "$REGION" \
  -c keyName="$KEY_NAME" \
  ${EXTRA_ARGS[@]:-} \
  "$@"
