#!/bin/bash
set -e

# Build and push Docker image to ECR
# Usage: ./scripts/build_and_push_docker.sh <ecr-repo-uri> <tag>

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <ecr-repo-uri> <tag>"
    echo "Example: $0 123456789012.dkr.ecr.us-east-1.amazonaws.com/compos3d v1.0.0"
    exit 1
fi

ECR_REPO=$1
TAG=$2

# Get AWS account and region from ECR URI
AWS_REGION=$(echo $ECR_REPO | cut -d'.' -f4)

echo "Building Docker image..."
docker build -t compos3d:$TAG .

echo "Tagging image for ECR..."
docker tag compos3d:$TAG $ECR_REPO:$TAG
docker tag compos3d:$TAG $ECR_REPO:latest

echo "Logging into ECR..."
aws ecr get-login-password --region $AWS_REGION | \
    docker login --username AWS --password-stdin $ECR_REPO

echo "Pushing image to ECR..."
docker push $ECR_REPO:$TAG
docker push $ECR_REPO:latest

echo "Done! Image available at:"
echo "  $ECR_REPO:$TAG"
echo "  $ECR_REPO:latest"
