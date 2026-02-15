# AWS provider for Compos3D Terraform

terraform {
  required_version = ">= 1.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region

  # Credentials from AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY env vars (e.g. from .env via scripts/terraform.ps1)
  # Or from shared config: aws configure
  default_tags {
    tags = {
      Project     = "Compos3D"
      Environment = "dev"
    }
  }
}

variable "aws_region" {
  description = "AWS region for dev resources"
  type        = string
  default     = "us-east-1"
}
