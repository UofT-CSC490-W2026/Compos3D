output "role_arn" {
  description = "ARN of the IAM role for Modal S3 access"
  value       = aws_iam_role.modal_s3_access.arn
}

output "role_name" {
  description = "Name of the IAM role"
  value       = aws_iam_role.modal_s3_access.name
}

output "modal_user_arn" {
  description = "ARN of the IAM user for Modal (when create_modal_user is true). Create access keys in AWS Console."
  value       = var.create_modal_user ? aws_iam_user.modal[0].arn : null
}

output "modal_user_name" {
  description = "Name of the IAM user for Modal (when create_modal_user is true)"
  value       = var.create_modal_user ? aws_iam_user.modal[0].name : null
}
