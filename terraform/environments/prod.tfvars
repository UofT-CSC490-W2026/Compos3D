project_name = "compos3d"
environment  = "prod"
aws_region   = "us-east-1"

# S3 bucket names
bronze_bucket = "compos3d-prod-bronze"
silver_bucket = "compos3d-prod-silver"
gold_bucket   = "compos3d-prod-gold"

# Glue catalog
glue_database_name = "compos3d_prod"
glue_crawler_name  = "compos3d_prod-silver-crawler"

# Athena
athena_workgroup_name = "compos3d_prod-workgroup"

# Tags
tags = {
  Environment = "prod"
  Project     = "compos3d"
  ManagedBy   = "terraform"
  CriticalData = "true"
}
