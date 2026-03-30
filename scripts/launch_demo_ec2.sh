#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <env> [experiment_name] [image_tag] [instance_type] [aws_region]" >&2
  echo "  env              dev | staging | prod" >&2
  echo "  experiment_name  defaults to claude_qwen" >&2
  echo "  image_tag        defaults to latest" >&2
  echo "  instance_type    defaults to g5.xlarge (GPU required for Blender rendering)" >&2
  echo "  aws_region       defaults to us-east-1" >&2
  exit 1
fi

ENV="$1"
EXPERIMENT_NAME="${2:-claude_qwen}"
IMAGE_TAG="${3:-latest}"
INSTANCE_TYPE="${4:-g5.xlarge}"
AWS_REGION="${5:-us-east-1}"

ECR_REPOSITORY_URL=$(terraform -chdir=terraform output -raw ecr_repository_url)
INSTANCE_PROFILE=$(terraform -chdir=terraform output -raw ec2_instance_profile_name)
SUBNET_ID=$(terraform -chdir=terraform output -raw ec2_primary_subnet_id)
SECURITY_GROUP_ID=$(terraform -chdir=terraform output -raw ec2_security_group_id)
GOLD_BUCKET=$(terraform -chdir=terraform output -raw gold_bucket)
ANTHROPIC_SECRET=$(terraform -chdir=terraform output -raw anthropic_secret_name)

ECR_REGISTRY="${ECR_REPOSITORY_URL%/*}"
IMAGE_REF="${ECR_REPOSITORY_URL}:demo-${IMAGE_TAG}"

# hypothesis_bank lives in gold under gold/hypothesis_banks/<experiment_name>/
BANK_S3_URI="s3://${GOLD_BUCKET}/compos3d/gold/hypothesis_banks/${EXPERIMENT_NAME}/hypothesis_bank.json"

VPC_ID=$(aws ec2 describe-security-groups \
  --group-ids "$SECURITY_GROUP_ID" \
  --query 'SecurityGroups[0].VpcId' --output text \
  --region "$AWS_REGION")

DEMO_SG=$(aws ec2 create-security-group \
  --group-name "compos3d-${ENV}-demo-sg-$(date +%s)" \
  --description "Compos3D demo inbound access" \
  --vpc-id "$VPC_ID" \
  --region "$AWS_REGION" \
  --query 'GroupId' --output text)

aws ec2 authorize-security-group-ingress \
  --group-id "$DEMO_SG" \
  --protocol tcp --port 7860 --cidr 0.0.0.0/0 \
  --region "$AWS_REGION"

aws ec2 authorize-security-group-ingress \
  --group-id "$DEMO_SG" \
  --protocol tcp --port 22 --cidr 0.0.0.0/0 \
  --region "$AWS_REGION"

AMI_ID=$(aws ec2 describe-images \
  --owners amazon \
  --filters \
    "Name=name,Values=al2023-ami-2023*-x86_64" \
    "Name=state,Values=available" \
  --query 'sort_by(Images, &CreationDate)[-1].ImageId' \
  --output text \
  --region "$AWS_REGION")

# Secrets Manager injects ANTHROPIC_API_KEY at runtime so Bedrock calls work.
# The EC2 instance role already grants bedrock:InvokeModel via the IAM module,
# so no additional AWS credential plumbing is needed inside the container.
USER_DATA=$(cat <<EOF
#!/bin/bash
set -euo pipefail
yum install -y docker
systemctl start docker
aws ecr get-login-password --region ${AWS_REGION} | \
  docker login --username AWS --password-stdin ${ECR_REGISTRY}
docker pull ${IMAGE_REF}
ANTHROPIC_API_KEY=\$(aws secretsmanager get-secret-value \
  --secret-id ${ANTHROPIC_SECRET} \
  --region ${AWS_REGION} \
  --query SecretString --output text 2>/dev/null || echo "")
docker run -d \
  --restart unless-stopped \
  -p 7860:7860 \
  --gpus all \
  -e AWS_DEFAULT_REGION=${AWS_REGION} \
  -e COMPOS3D_ENV=${ENV} \
  -e COMPOS3D_BANK_S3_URI=${BANK_S3_URI} \
  -e ANTHROPIC_API_KEY="\${ANTHROPIC_API_KEY}" \
  --name compos3d-demo \
  ${IMAGE_REF}
EOF
)

INSTANCE_ID=$(aws ec2 run-instances \
  --image-id "$AMI_ID" \
  --instance-type "$INSTANCE_TYPE" \
  --iam-instance-profile Name="$INSTANCE_PROFILE" \
  --subnet-id "$SUBNET_ID" \
  --security-group-ids "$SECURITY_GROUP_ID" "$DEMO_SG" \
  --associate-public-ip-address \
  --user-data "$USER_DATA" \
  --tag-specifications \
    "ResourceType=instance,Tags=[{Key=Name,Value=compos3d-${ENV}-demo},{Key=Project,Value=compos3d},{Key=Environment,Value=${ENV}}]" \
  --region "$AWS_REGION" \
  --query 'Instances[0].InstanceId' \
  --output text)

echo "Launched instance: $INSTANCE_ID"
echo "Waiting for public IP..."
sleep 10

PUBLIC_IP=$(aws ec2 describe-instances \
  --instance-ids "$INSTANCE_ID" \
  --query 'Reservations[0].Instances[0].PublicIpAddress' \
  --output text \
  --region "$AWS_REGION")

echo ""
echo "Instance:   $INSTANCE_ID"
echo "Demo URL:   http://${PUBLIC_IP}:7860"
echo "Bank:       ${BANK_S3_URI}"
echo ""
echo "The container pulls the bank from S3 and starts the app (~2 min after instance is running)."
echo ""
echo "To tear down:"
echo "  aws ec2 terminate-instances --instance-ids $INSTANCE_ID --region $AWS_REGION"
echo "  aws ec2 delete-security-group --group-id $DEMO_SG --region $AWS_REGION"
