variable "project_name" { type = string }
variable "environment" { type = string }

# Secrets Manager for API keys and credentials
resource "aws_secretsmanager_secret" "openai_api_key" {
  name = "${var.project_name}-${var.environment}-openai-key"
  description = "OpenAI API key for VIGA agent"
  
  recovery_window_in_days = 7
}

resource "aws_secretsmanager_secret" "anthropic_api_key" {
  name = "${var.project_name}-${var.environment}-anthropic-key"
  description = "Anthropic API key for Claude"
  
  recovery_window_in_days = 7
}

resource "aws_secretsmanager_secret" "anyscale_api_key" {
  name = "${var.project_name}-${var.environment}-anyscale-key"
  description = "Anyscale API key for Ray clusters"
  
  recovery_window_in_days = 7
}

resource "aws_secretsmanager_secret" "wandb_api_key" {
  name = "${var.project_name}-${var.environment}-wandb-key"
  description = "Weights & Biases API key for experiment tracking"
  
  recovery_window_in_days = 7
}

# IAM policy to read secrets
resource "aws_iam_policy" "read_secrets" {
  name = "${var.project_name}-${var.environment}-read-secrets"
  
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "secretsmanager:GetSecretValue",
          "secretsmanager:DescribeSecret"
        ]
        Resource = [
          aws_secretsmanager_secret.openai_api_key.arn,
          aws_secretsmanager_secret.anthropic_api_key.arn,
          aws_secretsmanager_secret.anyscale_api_key.arn,
          aws_secretsmanager_secret.wandb_api_key.arn,
        ]
      }
    ]
  })
}

output "secrets_policy_arn" { value = aws_iam_policy.read_secrets.arn }
output "openai_secret_arn" { value = aws_secretsmanager_secret.openai_api_key.arn }
output "anthropic_secret_arn" { value = aws_secretsmanager_secret.anthropic_api_key.arn }
output "anyscale_secret_arn" { value = aws_secretsmanager_secret.anyscale_api_key.arn }
output "wandb_secret_arn" { value = aws_secretsmanager_secret.wandb_api_key.arn }
