variable "database_name" {
  description = "Name of the Glue Catalog database for Scene Graphs"
  type        = string
}

variable "tags" {
  description = "Additional tags for the Glue database"
  type        = map(string)
  default     = {}
}
