# AWS provider for Compos3D Terraform (prod)

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

  default_tags {
    tags = {
      Project     = "Compos3D"
      Environment = "prod"
    }
  }
}

variable "aws_region" {
  description = "AWS region for prod resources"
  type        = string
  default     = "us-east-1"
}
