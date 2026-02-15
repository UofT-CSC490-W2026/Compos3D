# IAM Roles for Anyscale Data Pipelines - Enterprise Setup
# Each pipeline gets its own role with least-privilege access

# ==============================================================================
# 1. BRONZE INGESTION PIPELINE ROLE
# ==============================================================================
resource "aws_iam_role" "anyscale_bronze_pipeline" {
  name = "${var.project_name}-${var.environment}-anyscale-bronze"
  
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ec2.amazonaws.com"
        }
      }
    ]
  })
  
  tags = {
    Name        = "${var.project_name}-${var.environment}-anyscale-bronze"
    Environment = var.environment
    Pipeline    = "bronze-ingestion"
  }
}

resource "aws_iam_role_policy" "bronze_pipeline_policy" {
  name = "${var.project_name}-${var.environment}-bronze-policy"
  role = aws_iam_role.anyscale_bronze_pipeline.id
  
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "BronzeS3Write"
        Effect = "Allow"
        Action = [
          "s3:PutObject",
          "s3:PutObjectAcl",
          "s3:ListBucket"
        ]
        Resource = [
          "arn:aws:s3:::${var.bronze_bucket}",
          "arn:aws:s3:::${var.bronze_bucket}/*"
        ]
      },
      {
        Sid    = "GlueCatalogWrite"
        Effect = "Allow"
        Action = [
          "glue:GetDatabase",
          "glue:GetTable",
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
        Sid    = "CloudWatchLogs"
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ]
        Resource = "arn:aws:logs:*:*:log-group:/anyscale/bronze-pipeline*"
      }
    ]
  })
}

# Attach secrets read policy
resource "aws_iam_role_policy_attachment" "bronze_secrets" {
  role       = aws_iam_role.anyscale_bronze_pipeline.name
  policy_arn = var.secrets_policy_arn
}

# Instance profile for Bronze pipeline
resource "aws_iam_instance_profile" "bronze_pipeline" {
  name = "${var.project_name}-${var.environment}-bronze-profile"
  role = aws_iam_role.anyscale_bronze_pipeline.name
}

# ==============================================================================
# 2. SILVER TRANSFORMATION PIPELINE ROLE
# ==============================================================================
resource "aws_iam_role" "anyscale_silver_pipeline" {
  name = "${var.project_name}-${var.environment}-anyscale-silver"
  
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ec2.amazonaws.com"
        }
      }
    ]
  })
  
  tags = {
    Name        = "${var.project_name}-${var.environment}-anyscale-silver"
    Environment = var.environment
    Pipeline    = "silver-transformation"
  }
}

resource "aws_iam_role_policy" "silver_pipeline_policy" {
  name = "${var.project_name}-${var.environment}-silver-policy"
  role = aws_iam_role.anyscale_silver_pipeline.id
  
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "BronzeS3Read"
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:ListBucket"
        ]
        Resource = [
          "arn:aws:s3:::${var.bronze_bucket}",
          "arn:aws:s3:::${var.bronze_bucket}/*"
        ]
      },
      {
        Sid    = "SilverS3Write"
        Effect = "Allow"
        Action = [
          "s3:PutObject",
          "s3:PutObjectAcl",
          "s3:ListBucket"
        ]
        Resource = [
          "arn:aws:s3:::${var.silver_bucket}",
          "arn:aws:s3:::${var.silver_bucket}/*"
        ]
      },
      {
        Sid    = "GlueCatalogReadWrite"
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
        Sid    = "CloudWatchLogs"
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ]
        Resource = "arn:aws:logs:*:*:log-group:/anyscale/silver-pipeline*"
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "silver_secrets" {
  role       = aws_iam_role.anyscale_silver_pipeline.name
  policy_arn = var.secrets_policy_arn
}

resource "aws_iam_instance_profile" "silver_pipeline" {
  name = "${var.project_name}-${var.environment}-silver-profile"
  role = aws_iam_role.anyscale_silver_pipeline.name
}

# ==============================================================================
# 3. GOLD AGGREGATION PIPELINE ROLE
# ==============================================================================
resource "aws_iam_role" "anyscale_gold_pipeline" {
  name = "${var.project_name}-${var.environment}-anyscale-gold"
  
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ec2.amazonaws.com"
        }
      }
    ]
  })
  
  tags = {
    Name        = "${var.project_name}-${var.environment}-anyscale-gold"
    Environment = var.environment
    Pipeline    = "gold-aggregation"
  }
}

resource "aws_iam_role_policy" "gold_pipeline_policy" {
  name = "${var.project_name}-${var.environment}-gold-policy"
  role = aws_iam_role.anyscale_gold_pipeline.id
  
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "SilverS3Read"
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:ListBucket"
        ]
        Resource = [
          "arn:aws:s3:::${var.silver_bucket}",
          "arn:aws:s3:::${var.silver_bucket}/*"
        ]
      },
      {
        Sid    = "GoldS3Write"
        Effect = "Allow"
        Action = [
          "s3:PutObject",
          "s3:PutObjectAcl",
          "s3:ListBucket"
        ]
        Resource = [
          "arn:aws:s3:::${var.gold_bucket}",
          "arn:aws:s3:::${var.gold_bucket}/*"
        ]
      },
      {
        Sid    = "GlueCatalogReadWrite"
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
        Sid    = "CloudWatchLogs"
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ]
        Resource = "arn:aws:logs:*:*:log-group:/anyscale/gold-pipeline*"
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "gold_secrets" {
  role       = aws_iam_role.anyscale_gold_pipeline.name
  policy_arn = var.secrets_policy_arn
}

resource "aws_iam_instance_profile" "gold_pipeline" {
  name = "${var.project_name}-${var.environment}-gold-profile"
  role = aws_iam_role.anyscale_gold_pipeline.name
}

# ==============================================================================
# OUTPUTS
# ==============================================================================
output "bronze_pipeline_role_arn" {
  value       = aws_iam_role.anyscale_bronze_pipeline.arn
  description = "ARN of the Bronze ingestion pipeline IAM role"
}

output "silver_pipeline_role_arn" {
  value       = aws_iam_role.anyscale_silver_pipeline.arn
  description = "ARN of the Silver transformation pipeline IAM role"
}

output "gold_pipeline_role_arn" {
  value       = aws_iam_role.anyscale_gold_pipeline.arn
  description = "ARN of the Gold aggregation pipeline IAM role"
}

output "bronze_instance_profile_arn" {
  value       = aws_iam_instance_profile.bronze_pipeline.arn
  description = "ARN of the Bronze pipeline instance profile"
}

output "silver_instance_profile_arn" {
  value       = aws_iam_instance_profile.silver_pipeline.arn
  description = "ARN of the Silver pipeline instance profile"
}

output "gold_instance_profile_arn" {
  value       = aws_iam_instance_profile.gold_pipeline.arn
  description = "ARN of the Gold pipeline instance profile"
}
