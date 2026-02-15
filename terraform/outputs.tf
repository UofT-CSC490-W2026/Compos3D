output "bronze_bucket" {
  description = "Bronze layer S3 bucket name"
  value       = module.s3_lake.bronze_bucket
}

output "silver_bucket" {
  description = "Silver layer S3 bucket name"
  value       = module.s3_lake.silver_bucket
}

output "gold_bucket" {
  description = "Gold layer S3 bucket name"
  value       = module.s3_lake.gold_bucket
}

output "glue_database_name" {
  description = "Glue Data Catalog database name"
  value       = module.glue_catalog.database_name
}

output "glue_crawler_name" {
  description = "Glue crawler name for silver tables"
  value       = module.glue_catalog.crawler_name
}

# Commented out - batch module not deployed yet
# output "batch_job_role_arn" {
#   description = "IAM role ARN for AWS Batch jobs"
#   value       = module.iam.batch_job_role_arn
# }

output "athena_workgroup" {
  description = "Athena workgroup name"
  value       = module.glue_catalog.athena_workgroup
}

# Commented out - ECR and Batch modules not deployed yet
# output "ecr_repository_url" {
#   description = "URL of the ECR repository"
#   value       = module.ecr.repository_url
# }
# 
# output "batch_compute_environment_arn" {
#   description = "ARN of the Batch Compute Environment"
#   value       = module.batch.compute_environment_arn
# }
# 
# output "batch_job_queue_arn" {
#   description = "ARN of the Batch Job Queue"
#   value       = module.batch.job_queue_arn
# }

output "s3_uris" {
  description = "S3 URIs for quick reference"
  value = {
    bronze = "s3://${module.s3_lake.bronze_bucket}/compos3d/"
    silver = "s3://${module.s3_lake.silver_bucket}/compos3d/"
    gold   = "s3://${module.s3_lake.gold_bucket}/compos3d/"
  }
}
