project_name = "compos3d"
environment  = "staging"
aws_region   = "us-east-1"

# S3 bucket names
bronze_bucket = "compos3d-staging-bronze"
silver_bucket = "compos3d-staging-silver"
gold_bucket   = "compos3d-staging-gold"

# Glue catalog
glue_database_name = "compos3d_staging"
glue_crawler_name  = "compos3d_staging-silver-crawler"

# Athena
athena_workgroup_name = "compos3d_staging-workgroup"

# Tags
tags = {
  Environment = "staging"
  Project     = "compos3d"
  ManagedBy   = "terraform"
}
