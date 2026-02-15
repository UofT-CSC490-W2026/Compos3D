# Compos3D Prod Environment - wires data_lake, iam, and schema modules

locals {
  env_prefix  = "prod"
  bucket_name = "${local.env_prefix}-compos3d-data-lake"
  role_name   = "${local.env_prefix}-compos3d-modal-s3"
  db_name     = "${local.env_prefix}_compos3d_scene_graphs"
}

module "data_lake" {
  source = "../../modules/data_lake"

  bucket_name       = local.bucket_name
  environment       = local.env_prefix
  enable_versioning = true # keep history in prod

  tags = {
    ManagedBy = "terraform"
  }
}

module "iam" {
  source = "../../modules/iam"

  role_name            = local.role_name
  data_lake_bucket_arn = module.data_lake.bucket_arn
  create_modal_user    = true
  modal_user_name      = "${local.env_prefix}-compos3d-modal-s3-user"

  tags = {
    ManagedBy = "terraform"
  }
}

module "schema" {
  source = "../../modules/schema"

  database_name = local.db_name

  tags = {
    ManagedBy = "terraform"
  }
}
