variable "project_name" { type = string }
variable "environment" { type = string }
variable "aws_region" { type = string }
variable "bronze_bucket" { type = string }
variable "silver_bucket" { type = string }
variable "gold_bucket" { type = string }
variable "secrets_policy_arn" { type = string }
variable "ecr_repository_arn" { type = string }
variable "log_group_name" { type = string }

data "aws_availability_zones" "available" {
  state = "available"
}

locals {
  vpc_cidr_by_env = {
    dev     = "10.40.0.0/16"
    staging = "10.41.0.0/16"
    prod    = "10.42.0.0/16"
  }
  vpc_cidr = lookup(local.vpc_cidr_by_env, var.environment, "10.49.0.0/16")
  az_count = min(2, length(data.aws_availability_zones.available.names))
  az_names = slice(data.aws_availability_zones.available.names, 0, local.az_count)
}

resource "aws_vpc" "job" {
  cidr_block           = local.vpc_cidr
  enable_dns_support   = true
  enable_dns_hostnames = true

  tags = {
    Name = "${var.project_name}-${var.environment}-vpc"
    Tier = "network"
  }
}

resource "aws_internet_gateway" "job" {
  vpc_id = aws_vpc.job.id

  tags = {
    Name = "${var.project_name}-${var.environment}-igw"
    Tier = "network"
  }
}

resource "aws_subnet" "public" {
  count = local.az_count

  vpc_id                  = aws_vpc.job.id
  availability_zone       = local.az_names[count.index]
  cidr_block              = cidrsubnet(local.vpc_cidr, 8, count.index)
  map_public_ip_on_launch = true

  tags = {
    Name = "${var.project_name}-${var.environment}-public-${substr(local.az_names[count.index], -1, 1)}"
    Tier = "public"
  }
}

resource "aws_route_table" "public" {
  vpc_id = aws_vpc.job.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.job.id
  }

  tags = {
    Name = "${var.project_name}-${var.environment}-public-rt"
    Tier = "public"
  }
}

resource "aws_route_table_association" "public" {
  count = local.az_count

  subnet_id      = aws_subnet.public[count.index].id
  route_table_id = aws_route_table.public.id
}

resource "aws_iam_role" "ec2_job" {
  name = "${var.project_name}-${var.environment}-ec2-job"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action    = "sts:AssumeRole"
      Effect    = "Allow"
      Principal = { Service = "ec2.amazonaws.com" }
    }]
  })
}

resource "aws_iam_role_policy_attachment" "ec2_secrets" {
  role       = aws_iam_role.ec2_job.name
  policy_arn = var.secrets_policy_arn
}

resource "aws_iam_role_policy" "ec2_runtime" {
  name = "${var.project_name}-${var.environment}-ec2-runtime"
  role = aws_iam_role.ec2_job.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "LakeS3Access"
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:ListBucket",
          "s3:DeleteObject"
        ]
        Resource = [
          "arn:aws:s3:::${var.bronze_bucket}",
          "arn:aws:s3:::${var.bronze_bucket}/*",
          "arn:aws:s3:::${var.silver_bucket}",
          "arn:aws:s3:::${var.silver_bucket}/*",
          "arn:aws:s3:::${var.gold_bucket}",
          "arn:aws:s3:::${var.gold_bucket}/*"
        ]
      },
      {
        Sid    = "CloudWatchLogs"
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents",
          "logs:DescribeLogStreams"
        ]
        Resource = "*"
      },
      {
        Sid    = "EcrPull"
        Effect = "Allow"
        Action = [
          "ecr:BatchGetImage",
          "ecr:BatchCheckLayerAvailability",
          "ecr:GetDownloadUrlForLayer"
        ]
        Resource = [var.ecr_repository_arn]
      },
      {
        Sid    = "EcrAuth"
        Effect = "Allow"
        Action = [
          "ecr:GetAuthorizationToken"
        ]
        Resource = "*"
      },
      {
        Sid    = "BedrockInvoke"
        Effect = "Allow"
        Action = [
          "bedrock:InvokeModel",
          "bedrock:InvokeModelWithResponseStream"
        ]
        Resource = "arn:aws:bedrock:${var.aws_region}::foundation-model/*"
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "ec2_ssm" {
  role       = aws_iam_role.ec2_job.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

resource "aws_iam_instance_profile" "ec2_job" {
  name = "${var.project_name}-${var.environment}-ec2-job"
  role = aws_iam_role.ec2_job.name
}

resource "aws_security_group" "ec2_job" {
  name        = "${var.project_name}-${var.environment}-ec2-job"
  description = "compos3d EC2 job nodes - egress only"
  vpc_id      = aws_vpc.job.id

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
    description = "Allow all outbound"
  }

  tags = {
    Name = "${var.project_name}-${var.environment}-ec2-job"
    Tier = "compute"
  }
}

resource "aws_cloudwatch_log_group" "jobs" {
  name              = var.log_group_name
  retention_in_days = 30
}

output "instance_profile_name" {
  description = "Value for ec2_iam_instance_profile in AppConfig"
  value       = aws_iam_instance_profile.ec2_job.name
}

output "vpc_id" {
  description = "VPC id for the Compos3D job network"
  value       = aws_vpc.job.id
}

output "primary_subnet_id" {
  description = "Primary public subnet id for EC2 job runners"
  value       = aws_subnet.public[0].id
}

output "public_subnet_ids" {
  description = "Public subnet ids for EC2 job runners"
  value       = aws_subnet.public[*].id
}

output "security_group_id" {
  description = "Value for ec2_security_group_id in AppConfig"
  value       = aws_security_group.ec2_job.id
}

output "ec2_job_role_arn" {
  value = aws_iam_role.ec2_job.arn
}

output "log_group_name" {
  value = aws_cloudwatch_log_group.jobs.name
}
