variable "role_name" {
  description = "Name of the IAM role for Modal S3 access"
  type        = string
}

variable "data_lake_bucket_arn" {
  description = "ARN of the S3 bucket (from data_lake module) to grant read/write access"
  type        = string
}

variable "trusted_principal_arn" {
  description = "Optional ARN of principal allowed to assume this role (defaults to current account)"
  type        = string
  default     = null
}

variable "create_modal_user" {
  description = "Create an IAM user for Modal with same S3 policy (create access keys in Console and add to Modal.Secret)"
  type        = bool
  default     = false
}

variable "modal_user_name" {
  description = "Name of the IAM user for Modal (when create_modal_user is true)"
  type        = string
  default     = "compos3d-modal-s3"
}

variable "tags" {
  description = "Additional tags for the role"
  type        = map(string)
  default     = {}
}
