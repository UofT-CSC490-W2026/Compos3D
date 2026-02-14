# AWS Glue Catalog Database for Compos3D Scene Graphs (metadata about object positions)

resource "aws_glue_catalog_database" "scene_graphs" {
  name        = var.database_name
  description = "Schema for Compos3D Scene Graphs - metadata about object positions in 3D scenes"

  create_table_default_permission {
    permissions = ["ALL"]
    principal {
      data_lake_principal_identifier = "IAM_ALLOWED_PRINCIPALS"
    }
  }
}
