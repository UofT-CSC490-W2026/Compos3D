bucket         = "compos3d-terraform-state"
key            = "compos3d/prod/terraform.tfstate"
region         = "us-east-1"
dynamodb_table = "compos3d-terraform-locks"
encrypt        = true
