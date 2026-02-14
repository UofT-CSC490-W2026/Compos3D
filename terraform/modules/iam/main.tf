# IAM Role for Modal functions - read/write access to Compos3D Data Lake S3 bucket

data "aws_caller_identity" "current" {}

resource "aws_iam_role" "modal_s3_access" {
  name               = var.role_name
  description        = "IAM role for Modal functions to read/write Compos3D Data Lake S3 bucket"
  assume_role_policy  = data.aws_iam_policy_document.assume_role.json

  tags = merge(var.tags, {
    Project = "Compos3D"
  })
}

# Allow Modal (or any principal) to assume this role when using long-lived credentials
# In practice, Modal uses AWS keys (from Secrets) so this role is assumed by whoever holds those keys.
# This policy allows the current AWS account to assume the role (e.g. for testing).
# For Modal: you create IAM user keys with this role's policy attached, or attach inline policy to the user.
data "aws_iam_policy_document" "assume_role" {
  statement {
    effect = "Allow"
    principals {
      type        = "AWS"
      identifiers = [var.trusted_principal_arn != null ? var.trusted_principal_arn : data.aws_caller_identity.current.account_id]
    }
    actions = ["sts:AssumeRole"]
  }
}

resource "aws_iam_role_policy" "s3_data_lake" {
  name   = "${var.role_name}-s3-data-lake"
  role   = aws_iam_role.modal_s3_access.id
  policy = data.aws_iam_policy_document.s3_data_lake.json
}

data "aws_iam_policy_document" "s3_data_lake" {
  statement {
    sid    = "ListBucket"
    effect = "Allow"
    actions = [
      "s3:ListBucket",
      "s3:GetBucketLocation"
    ]
    resources = [var.data_lake_bucket_arn]
  }

  statement {
    sid    = "ReadWriteObjects"
    effect = "Allow"
    actions = [
      "s3:GetObject",
      "s3:PutObject",
      "s3:DeleteObject",
      "s3:GetObjectVersion",
      "s3:PutObjectAcl",
      "s3:GetObjectTagging",
      "s3:PutObjectTagging",
      "s3:DeleteObjectTagging"
    ]
    resources = ["${var.data_lake_bucket_arn}/*"]
  }
}

# Optional IAM user for Modal: create access keys in AWS Console and add to Modal.Secret
resource "aws_iam_user" "modal" {
  count = var.create_modal_user ? 1 : 0

  name = var.modal_user_name
  path = "/compos3d/"

  tags = merge(var.tags, {
    Project = "Compos3D"
    Purpose = "Modal S3 access"
  })
}

resource "aws_iam_user_policy" "modal_s3" {
  count = var.create_modal_user ? 1 : 0

  name   = "s3-data-lake"
  user   = aws_iam_user.modal[0].name
  policy = data.aws_iam_policy_document.s3_data_lake.json
}
