variable "project_name" {
  description = "Project name for resource naming"
  type        = string
  default     = "compos3d"
}

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
}

variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "bronze_bucket_suffix" {
  description = "Suffix for bronze bucket"
  type        = string
  default     = "bronze"
}

variable "silver_bucket_suffix" {
  description = "Suffix for silver bucket"
  type        = string
  default     = "silver"
}

variable "gold_bucket_suffix" {
  description = "Suffix for gold bucket"
  type        = string
  default     = "gold"
}

variable "tags" {
  description = "Common tags for all resources"
  type        = map(string)
  default     = {}
}
