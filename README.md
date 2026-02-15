# Compos3D Data Platform

Data engineering infrastructure for Compositional 3D Scene Generation.

## Setup

```bash
# Install dependencies
uv sync

# Configure AWS SSO
aws configure sso
# Enter: https://d-9d675989d2.awsapps.com/start
# Profile: myisb_IsbUsersPS-136268833180

# Deploy infrastructure
cd terraform
terraform init
terraform apply -var-file=environments/dev.tfvars

# Set environment
export AWS_PROFILE=myisb_IsbUsersPS-136268833180
export COMPOS3D_S3_BUCKET=compos3d-dev-silver
```

## Usage

```bash
# Ingest data to Bronze
uv run python -m compos3d_dp.cli bronze-demo --config-path config/env.aws.yaml

# Transform Bronze → Silver
uv run python -m compos3d_dp.cli silver-build --config-path config/env.aws.yaml

# Discover tables in Glue
aws glue start-crawler --name compos3d_dev-silver-crawler

# Query with Athena
aws athena start-query-execution \
  --query-string "SELECT * FROM scene LIMIT 10" \
  --query-execution-context Database=compos3d_dev \
  --result-configuration OutputLocation=s3://compos3d-dev-silver/athena-results/ \
  --work-group compos3d_dev-workgroup
```

## Structure

```
├── src/compos3d_dp/
│   ├── cli.py              # CLI entrypoint
│   ├── config.py           # Configuration management
│   ├── schemas/            # Pydantic data models
│   ├── pipelines/          # ETL pipelines
│   ├── storage/            # S3/Local storage
│   ├── compute/            # Modal integration
│   └── utils/              # Manifest & GE validation
├── terraform/              # AWS infrastructure
└── config/                 # Environment configs
```

## AWS Resources

- **S3**: Bronze/Silver/Gold buckets
- **Glue**: Data catalog + crawler
- **Athena**: SQL queries on data lake
- **IAM**: Access roles

## Modal (Distributed Compute)

```python
from compos3d_dp.compute.modal_runner import submit_batch_to_modal

# Process 1000s of scenes in parallel
submit_batch_to_modal(
    scene_ids=your_scenes,
    s3_bucket="compos3d-dev-silver",
    batch_size=100
)
```
