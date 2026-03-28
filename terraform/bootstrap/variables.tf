variable "aws_region" {
  type    = string
  default = "us-east-1"
}

variable "state_bucket_name" {
  type    = string
  default = "compos3d-terraform-state"
}

variable "lock_table_name" {
  type    = string
  default = "compos3d-terraform-locks"
}
