variable "project_name" { type = string }
variable "environment" { type = string }
variable "batch_service_role_arn" { type = string }
variable "batch_job_role_arn" { type = string }
variable "vpc_id" { 
  type = string 
  description = "VPC ID where Batch resources will run"
}
variable "subnet_ids" { 
  type = list(string) 
  description = "Subnets where Batch resources will run"
}
variable "security_group_ids" { 
  type = list(string) 
  description = "Security groups for Batch resources"
}

# Compute Environment (FARGATE for simplicity/speed)
resource "aws_batch_compute_environment" "fargate" {
  compute_environment_name = "${var.project_name}-${var.environment}-fargate"

  compute_resources {
    type = "FARGATE"
    max_vcpus = 16
    
    subnets = var.subnet_ids
    security_group_ids = var.security_group_ids
  }

  service_role = var.batch_service_role_arn
  type         = "MANAGED"
  state        = "ENABLED"
}

# Job Queue
resource "aws_batch_job_queue" "default" {
  name     = "${var.project_name}-${var.environment}-queue"
  state    = "ENABLED"
  priority = 1

  compute_environments = [
    aws_batch_compute_environment.fargate.arn,
  ]
}

output "compute_environment_arn" {
  value = aws_batch_compute_environment.fargate.arn
}

output "job_queue_arn" {
  value = aws_batch_job_queue.default.arn
}
