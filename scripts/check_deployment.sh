#!/bin/bash
# Check deployment readiness

echo "=== Compos3D Deployment Check ==="
echo

# Check for AWS_PROFILE
if [ -n "$AWS_PROFILE" ]; then
    echo "Using AWS Profile: $AWS_PROFILE"
    echo
fi

# Check AWS
echo "1. Checking AWS credentials..."
if aws sts get-caller-identity &>/dev/null; then
    echo "   ✅ AWS credentials configured"
    aws sts get-caller-identity | grep Account | cut -d'"' -f4 | xargs echo "   Account:"
else
    echo "   ❌ AWS not configured."
    echo "   For SSO: aws configure sso, then aws sso login --profile sandbox"
    echo "   For IAM: aws configure"
    exit 1
fi
echo

# Check Terraform
echo "2. Checking Terraform..."
if command -v terraform &>/dev/null; then
    echo "   ✅ Terraform installed"
    terraform version | head -1
else
    echo "   ❌ Terraform not installed"
    exit 1
fi
echo

# Check Modal
echo "3. Checking Modal..."
if modal token current &>/dev/null; then
    echo "   ✅ Modal authenticated"
else
    echo "   ⚠️  Modal not authenticated. Run: modal token new"
fi
echo

# Check if Terraform is deployed
echo "4. Checking AWS infrastructure..."
cd terraform 2>/dev/null
if [ -f terraform.tfstate ]; then
    echo "   ✅ Infrastructure deployed"
    terraform output -json | python3 -m json.tool | grep -A1 '"value"' | head -5
else
    echo "   ⚠️  Infrastructure not deployed. Run: terraform apply"
fi
cd ..
echo

echo "=== Next Steps ==="
if [ ! -f terraform/terraform.tfstate ]; then
    echo "1. Deploy AWS: cd terraform && terraform apply -var-file=environments/dev.tfvars"
    echo "2. Setup Modal: modal token new"
    echo "3. Run pipeline: See DEPLOY_NOW.md"
else
    echo "✅ Ready to use!"
    echo "Run: uv run python -m compos3d_dp.cli bronze-demo --config-path config/env.aws.yaml"
fi
