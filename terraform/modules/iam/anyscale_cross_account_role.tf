# Cross-Account IAM Role for Anyscale to Access Our AWS Resources
# This allows Anyscale jobs to assume into our AWS account

# Generate random external ID for additional security
resource "random_string" "external_id" {
  length  = 16
  special = false
}

# ==============================================================================
# CROSS-ACCOUNT ROLE - Anyscale can assume this from their AWS account
# ==============================================================================
resource "aws_iam_role" "anyscale_cross_account" {
  name = "${var.project_name}-${var.environment}-anyscale-access"
  
  # Trust policy - allows Anyscale's AWS identity to assume this role
  # NOTE: Update the Principal after running get_anyscale_identity.py
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          # Allow Anyscale's AWS account to assume this role
          # Account ID from get_anyscale_identity.py output
          AWS = "arn:aws:iam::471112779209:root"
        }
        Action = "sts:AssumeRole"
        Condition = {
          StringEquals = {
            "sts:ExternalId" = random_string.external_id.result
          }
        }
      }
    ]
  })
  
  tags = {
    Name        = "${var.project_name}-${var.environment}-anyscale-access"
    Environment = var.environment
    Purpose     = "cross-account-anyscale"
  }
}

# Policy for cross-account role - full data pipeline access
resource "aws_iam_role_policy" "anyscale_cross_account_policy" {
  name = "${var.project_name}-${var.environment}-anyscale-policy"
  role = aws_iam_role.anyscale_cross_account.id
  
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "S3DataLakeAccess"
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject",
          "s3:ListBucket"
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
        Sid    = "GlueCatalogAccess"
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
        Sid    = "SecretsManagerRead"
        Effect = "Allow"
        Action = [
          "secretsmanager:GetSecretValue",
          "secretsmanager:DescribeSecret"
        ]
        Resource = "arn:aws:secretsmanager:*:*:secret:${var.project_name}-${var.environment}-*"
      },
      {
        Sid    = "CloudWatchLogs"
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ]
        Resource = "arn:aws:logs:*:*:log-group:/anyscale/*"
      },
      {
        Sid    = "BedrockAccess"
        Effect = "Allow"
        Action = [
          "bedrock:InvokeModel",
          "bedrock:InvokeModelWithResponseStream"
        ]
        Resource = "arn:aws:bedrock:*::foundation-model/*"
      }
    ]
  })
}

# ==============================================================================
# OUTPUTS
# ==============================================================================
output "anyscale_cross_account_role_arn" {
  value       = aws_iam_role.anyscale_cross_account.arn
  description = "ARN of the cross-account role that Anyscale jobs should assume"
}

output "anyscale_external_id" {
  value       = random_string.external_id.result
  description = "External ID to use when assuming the cross-account role"
  sensitive   = true
}
