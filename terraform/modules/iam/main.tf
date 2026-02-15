variable "project_name" { type = string }
variable "environment" { type = string }
variable "bronze_bucket" { type = string }
variable "silver_bucket" { type = string }
variable "gold_bucket" { type = string }
variable "glue_db_name" { type = string }
variable "secrets_policy_arn" { type = string }

# IAM Role for AWS Batch jobs
resource "aws_iam_role" "batch_job" {
  name = "${var.project_name}-${var.environment}-batch-job"
  
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "ecs-tasks.amazonaws.com"
      }
    }]
  })
}

# Policy for Batch jobs to access S3 and Glue
# Attach secrets read policy to batch job role
resource "aws_iam_role_policy_attachment" "batch_secrets" {
  role       = aws_iam_role.batch_job.name
  policy_arn = var.secrets_policy_arn
}

resource "aws_iam_role_policy" "batch_job_policy" {
  name = "${var.project_name}-${var.environment}-batch-policy"
  role = aws_iam_role.batch_job.id
  
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
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
        Effect = "Allow"
        Action = [
          "glue:GetDatabase",
          "glue:GetTable",
          "glue:GetPartitions",
          "glue:CreateTable",
          "glue:UpdateTable",
          "glue:BatchCreatePartition"
        ]
        Resource = [
          "arn:aws:glue:*:*:catalog",
          "arn:aws:glue:*:*:database/${var.glue_db_name}",
          "arn:aws:glue:*:*:table/${var.glue_db_name}/*"
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ]
        Resource = "arn:aws:logs:*:*:*"
      }
    ]
  })
}

# IAM Role for AWS Batch service
resource "aws_iam_role" "batch_service" {
  name = "${var.project_name}-${var.environment}-batch-service"
  
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "batch.amazonaws.com"
      }
    }]
  })
}

resource "aws_iam_role_policy_attachment" "batch_service" {
  role       = aws_iam_role.batch_service.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSBatchServiceRole"
}

# IAM Role for Athena queries
resource "aws_iam_role" "athena_user" {
  name = "${var.project_name}-${var.environment}-athena-user"
  
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        AWS = data.aws_caller_identity.current.account_id
      }
    }]
  })
}

resource "aws_iam_role_policy" "athena_user_policy" {
  name = "${var.project_name}-${var.environment}-athena-policy"
  role = aws_iam_role.athena_user.id
  
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:ListBucket"
        ]
        Resource = [
          "arn:aws:s3:::${var.silver_bucket}",
          "arn:aws:s3:::${var.silver_bucket}/*",
          "arn:aws:s3:::${var.gold_bucket}",
          "arn:aws:s3:::${var.gold_bucket}/*"
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "s3:PutObject",
          "s3:GetObject"
        ]
        Resource = "arn:aws:s3:::${var.silver_bucket}/athena-results/*"
      },
      {
        Effect = "Allow"
        Action = [
          "athena:StartQueryExecution",
          "athena:GetQueryExecution",
          "athena:GetQueryResults",
          "athena:StopQueryExecution",
          "athena:GetWorkGroup"
        ]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "glue:GetDatabase",
          "glue:GetTable",
          "glue:GetPartitions"
        ]
        Resource = [
          "arn:aws:glue:*:*:catalog",
          "arn:aws:glue:*:*:database/${var.glue_db_name}",
          "arn:aws:glue:*:*:table/${var.glue_db_name}/*"
        ]
      }
    ]
  })
}

data "aws_caller_identity" "current" {}

output "batch_job_role_arn" { value = aws_iam_role.batch_job.arn }
output "batch_service_role_arn" { value = aws_iam_role.batch_service.arn }
output "athena_user_role_arn" { value = aws_iam_role.athena_user.arn }
