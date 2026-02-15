variable "database_name" { type = string }
variable "silver_bucket" { type = string }
variable "environment" { type = string }

# Glue Database
resource "aws_glue_catalog_database" "main" {
  name        = var.database_name
  description = "Compos3D data catalog for ${var.environment}"
}

# IAM role for Glue crawler
resource "aws_iam_role" "glue_crawler" {
  name = "${var.database_name}-crawler-role"
  
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "glue.amazonaws.com"
      }
    }]
  })
}

resource "aws_iam_role_policy_attachment" "glue_service" {
  role       = aws_iam_role.glue_crawler.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSGlueServiceRole"
}

resource "aws_iam_role_policy" "glue_s3_access" {
  name = "${var.database_name}-s3-access"
  role = aws_iam_role.glue_crawler.id
  
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
          "arn:aws:s3:::${var.silver_bucket}/*"
        ]
      }
    ]
  })
}

# Glue Crawler for Silver tables
resource "aws_glue_crawler" "silver" {
  name          = "${var.database_name}-silver-crawler"
  role          = aws_iam_role.glue_crawler.arn
  database_name = aws_glue_catalog_database.main.name
  
  s3_target {
    path = "s3://${var.silver_bucket}/compos3d/silver/"
  }
  
  configuration = jsonencode({
    Version = 1.0
    CrawlerOutput = {
      Partitions = { AddOrUpdateBehavior = "InheritFromTable" }
    }
    Grouping = {
      TableGroupingPolicy = "CombineCompatibleSchemas"
    }
  })
  
  schema_change_policy {
    delete_behavior = "LOG"
    update_behavior = "UPDATE_IN_DATABASE"
  }
  
  recrawl_policy {
    recrawl_behavior = "CRAWL_EVERYTHING"
  }
}

# Athena workgroup for queries
resource "aws_athena_workgroup" "main" {
  name = "${var.database_name}-workgroup"
  
  configuration {
    enforce_workgroup_configuration    = true
    publish_cloudwatch_metrics_enabled = true
    
    result_configuration {
      output_location = "s3://${var.silver_bucket}/athena-results/"
      
      encryption_configuration {
        encryption_option = "SSE_S3"
      }
    }
  }
}

output "database_name" { value = aws_glue_catalog_database.main.name }
output "crawler_name" { value = aws_glue_crawler.silver.name }
output "crawler_role_arn" { value = aws_iam_role.glue_crawler.arn }
output "athena_workgroup" { value = aws_athena_workgroup.main.name }
