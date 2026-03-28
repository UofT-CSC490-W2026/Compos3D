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

output "ecr_repository_url" {
  description = "URL of the Compos3D runtime ECR repository"
  value       = module.ecr.repository_url
}

output "ecr_repository_name" {
  description = "Name of the Compos3D runtime ECR repository"
  value       = module.ecr.repository_name
}

output "ec2_instance_profile_name" {
  description = "EC2 instance profile name for job runners"
  value       = module.ec2_compute.instance_profile_name
}

output "ec2_vpc_id" {
  description = "VPC id for Compos3D EC2 job runners"
  value       = module.ec2_compute.vpc_id
}

output "ec2_primary_subnet_id" {
  description = "Primary subnet id for Compos3D EC2 job runners"
  value       = module.ec2_compute.primary_subnet_id
}

output "ec2_public_subnet_ids" {
  description = "Public subnet ids for Compos3D EC2 job runners"
  value       = module.ec2_compute.public_subnet_ids
}

output "ec2_security_group_id" {
  description = "EC2 security group id for job runners"
  value       = module.ec2_compute.security_group_id
}

output "ec2_log_group_name" {
  description = "CloudWatch log group for EC2 job containers"
  value       = module.ec2_compute.log_group_name
}

output "openai_secret_arn" {
  value = module.secrets.openai_secret_arn
}

output "anthropic_secret_arn" {
  value = module.secrets.anthropic_secret_arn
}

output "anyscale_secret_arn" {
  value = module.secrets.anyscale_secret_arn
}

output "wandb_secret_arn" {
  value = module.secrets.wandb_secret_arn
}

output "openai_secret_name" {
  value = module.secrets.openai_secret_name
}

output "anthropic_secret_name" {
  value = module.secrets.anthropic_secret_name
}

output "anyscale_secret_name" {
  value = module.secrets.anyscale_secret_name
}

output "wandb_secret_name" {
  value = module.secrets.wandb_secret_name
}

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
