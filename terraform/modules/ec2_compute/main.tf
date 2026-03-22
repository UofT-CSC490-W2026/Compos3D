variable "project_name" { type = string }
variable "environment" { type = string }
variable "bronze_bucket" { type = string }
variable "silver_bucket" { type = string }
variable "gold_bucket" { type = string }

# ---------------------------------------------------------------------------
# IAM role that EC2 instances assume to access S3, SSM, and CloudWatch Logs.
# The instance profile is referenced as ec2_iam_instance_profile in the
# AppConfig / env YAML files.
# ---------------------------------------------------------------------------

resource "aws_iam_role" "ec2_job" {
  name = "${var.project_name}-${var.environment}-ec2-job"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action    = "sts:AssumeRole"
      Effect    = "Allow"
      Principal = { Service = "ec2.amazonaws.com" }
    }]
  })
}

# S3 read/write on all three lake buckets
resource "aws_iam_role_policy" "ec2_s3" {
  name = "${var.project_name}-${var.environment}-ec2-s3"
  role = aws_iam_role.ec2_job.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:ListBucket",
          "s3:DeleteObject"
        ]
        Resource = [
          "arn:aws:s3:::${var.bronze_bucket}",
          "arn:aws:s3:::${var.bronze_bucket}/*",
          "arn:aws:s3:::${var.silver_bucket}",
          "arn:aws:s3:::${var.silver_bucket}/*",
          "arn:aws:s3:::${var.gold_bucket}",
          "arn:aws:s3:::${var.gold_bucket}/*"
        ]
      }
    ]
  })
}

# CloudWatch Logs: write job output
resource "aws_iam_role_policy" "ec2_logs" {
  name = "${var.project_name}-${var.environment}-ec2-logs"
  role = aws_iam_role.ec2_job.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Action = [
        "logs:CreateLogGroup",
        "logs:CreateLogStream",
        "logs:PutLogEvents",
        "logs:DescribeLogStreams"
      ]
      Resource = "arn:aws:logs:*:*:*"
    }]
  })
}

# SSM: allow Session Manager and Run Command (no SSH key needed)
resource "aws_iam_role_policy_attachment" "ec2_ssm" {
  role       = aws_iam_role.ec2_job.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

# EC2 instance profile (wraps the role for assignment to instances)
resource "aws_iam_instance_profile" "ec2_job" {
  name = "${var.project_name}-${var.environment}-ec2-job"
  role = aws_iam_role.ec2_job.name
}

# ---------------------------------------------------------------------------
# Security group: outbound-only (HTTPS for pip/git, S3, SSM endpoints).
# No inbound rules — SSH is not needed when using SSM.
# ---------------------------------------------------------------------------

resource "aws_security_group" "ec2_job" {
  name        = "${var.project_name}-${var.environment}-ec2-job"
  description = "compos3d EC2 job nodes — egress only"

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
    description = "Allow all outbound"
  }
}

# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------

output "instance_profile_name" {
  description = "Value for ec2_iam_instance_profile in AppConfig"
  value       = aws_iam_instance_profile.ec2_job.name
}

output "security_group_id" {
  description = "Value for ec2_security_group_id in AppConfig"
  value       = aws_security_group.ec2_job.id
}

output "ec2_job_role_arn" {
  value = aws_iam_role.ec2_job.arn
}
