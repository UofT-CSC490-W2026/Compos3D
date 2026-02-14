output "database_name" {
  description = "Name of the Glue Catalog database"
  value       = aws_glue_catalog_database.scene_graphs.name
}

output "database_arn" {
  description = "ARN of the Glue Catalog database"
  value       = aws_glue_catalog_database.scene_graphs.arn
}
