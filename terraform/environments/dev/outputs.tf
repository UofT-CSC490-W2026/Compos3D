output "data_lake_bucket_name" {
  description = "S3 bucket name for 3D assets and Hypothesis Bank"
  value       = module.data_lake.bucket_id
}

output "data_lake_bucket_arn" {
  description = "S3 bucket ARN"
  value       = module.data_lake.bucket_arn
}

output "modal_iam_role_arn" {
  description = "IAM role ARN for Modal"
  value       = module.iam.role_arn
}

output "modal_iam_user_name" {
  description = "IAM user for Modal - create access keys in AWS Console and add to Modal Secret 'aws-compos3d'"
  value       = module.iam.modal_user_name
}

output "glue_database_name" {
  description = "Glue Catalog database name for Scene Graphs"
  value       = module.schema.database_name
}
