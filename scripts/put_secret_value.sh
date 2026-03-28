#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <secret_name_or_arn> <secret_value> [aws_region]" >&2
  exit 1
fi

SECRET_ID="$1"
SECRET_VALUE="$2"
AWS_REGION="${3:-us-east-1}"

aws secretsmanager put-secret-value \
  --secret-id "$SECRET_ID" \
  --secret-string "$SECRET_VALUE" \
  --region "$AWS_REGION"
