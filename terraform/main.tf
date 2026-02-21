terraform {
  required_version = ">= 1.5"
  
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
  
  # Uncomment for remote state
  # backend "s3" {
  #   bucket = "compos3d-terraform-state"
  #   key    = "compos3d/terraform.tfstate"
  #   region = "us-east-1"
  # }
}

provider "aws" {
  region = var.aws_region
  
  default_tags {
    tags = merge(
      {
        Project     = var.project_name
        Environment = var.environment
        ManagedBy   = "terraform"
      },
      var.tags
    )
  }
}

# Local variables
locals {
  bronze_bucket = "${var.project_name}-${var.environment}-${var.bronze_bucket_suffix}"
  silver_bucket = "${var.project_name}-${var.environment}-${var.silver_bucket_suffix}"
  gold_bucket   = "${var.project_name}-${var.environment}-${var.gold_bucket_suffix}"
  glue_db_name  = "${var.project_name}_${var.environment}"
}

# Data Lake S3 Buckets
module "s3_lake" {
  source = "./modules/s3_lake"
  
  bronze_bucket = local.bronze_bucket
  silver_bucket = local.silver_bucket
  gold_bucket   = local.gold_bucket
  environment   = var.environment
}

# Glue Data Catalog
module "glue_catalog" {
  source = "./modules/glue_catalog"
  
  database_name = local.glue_db_name
  silver_bucket = local.silver_bucket
  environment   = var.environment
  
  depends_on = [module.s3_lake]
}

# Secrets Manager
module "secrets" {
  source = "./modules/secrets"
  
  project_name = var.project_name
  environment  = var.environment
}

# IAM Roles and Policies
module "iam" {
  source = "./modules/iam"
  
  project_name  = var.project_name
  environment   = var.environment
  bronze_bucket = local.bronze_bucket
  silver_bucket = local.silver_bucket
  gold_bucket   = local.gold_bucket
  glue_db_name  = local.glue_db_name
  
  secrets_policy_arn = module.secrets.secrets_policy_arn
}

# Batch compute (commented out for initial deployment)
# Uncomment when you need distributed compute
# data "aws_vpc" "default" {
#   default = true
# }
# 
# data "aws_subnets" "default" {
#   filter {
#     name   = "vpc-id"
#     values = [data.aws_vpc.default.id]
#   }
# }
# 
# data "aws_security_group" "default" {
#   filter {
#     name   = "vpc-id"
#     values = [data.aws_vpc.default.id]
#   }
#   filter {
#     name   = "group-name"
#     values = ["default"]
#   }
# }
# 
# # ECR Repository
# module "ecr" {
#   source = "./modules/ecr"
#   
#   project_name = var.project_name
#   environment  = var.environment
# }
# 
# # AWS Batch
# module "batch" {
#   source = "./modules/batch"
#   
#   project_name     = var.project_name
#   environment      = var.environment
#   batch_job_role_arn     = module.iam.batch_job_role_arn
#   batch_service_role_arn = module.iam.batch_service_role_arn
#   
#   vpc_id             = data.aws_vpc.default.id
#   subnet_ids         = data.aws_subnets.default.ids
#   security_group_ids = [data.aws_security_group.default.id]
# }
