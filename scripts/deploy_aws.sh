#!/bin/bash
# Deploy Compos3D to AWS with SSO profile

set -e

# Your SSO profile
export AWS_PROFILE=myisb_IsbUsersPS-136268833180

echo "=== Deploying Compos3D to AWS ==="
echo "Using AWS Profile: $AWS_PROFILE"
echo

# Check credentials
echo "1. Checking AWS credentials..."
if ! aws sts get-caller-identity &>/dev/null; then
    echo "❌ SSO session expired. Logging in..."
    aws sso login --profile $AWS_PROFILE
fi

ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
echo "✅ Connected to AWS Account: $ACCOUNT"
echo

# Navigate to terraform directory
cd "$(dirname "$0")/../terraform"

# Terraform init
echo "2. Initializing Terraform..."
terraform init
echo

# Terraform plan
echo "3. Planning infrastructure..."
terraform plan -var-file=environments/dev.tfvars
echo

# Ask for confirmation
read -p "Do you want to apply these changes? (yes/no): " confirm
if [ "$confirm" != "yes" ]; then
    echo "Deployment cancelled."
    exit 0
fi

# Terraform apply
echo "4. Applying infrastructure..."
terraform apply -var-file=environments/dev.tfvars -auto-approve

# Get outputs
echo
echo "=== Deployment Complete! ==="
echo
terraform output

# Save bucket name for easy access
BUCKET=$(terraform output -raw silver_bucket)
echo
echo "To use the pipeline, run:"
echo "export AWS_PROFILE=$AWS_PROFILE"
echo "export COMPOS3D_S3_BUCKET=$BUCKET"
echo "uv run python -m compos3d_dp.cli silver-build --config-path config/env.aws.yaml"
