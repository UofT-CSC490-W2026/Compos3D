#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <ecr_repository_url> <image_tag> [aws_region]" >&2
  exit 1
fi

ECR_REPOSITORY_URL="$1"
IMAGE_TAG="$2"
AWS_REGION="${3:-us-east-1}"
IMAGE_REF="${ECR_REPOSITORY_URL}:${IMAGE_TAG}"
ECR_REGISTRY="${ECR_REPOSITORY_URL%/*}"

aws ecr get-login-password --region "$AWS_REGION" | \
  docker login --username AWS --password-stdin "$ECR_REGISTRY"

docker build -t "$IMAGE_REF" .
docker push "$IMAGE_REF"

echo "Pushed runtime image: $IMAGE_REF"
