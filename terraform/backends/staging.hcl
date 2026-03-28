bucket         = "compos3d-terraform-state"
key            = "compos3d/staging/terraform.tfstate"
region         = "us-east-1"
dynamodb_table = "compos3d-terraform-locks"
encrypt        = true
