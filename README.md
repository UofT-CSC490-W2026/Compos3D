# Compos3D

3D scene generation system.

## Setup

### Environment Setup

Install dependencies with uv:

```bash
uv venv
source .venv/bin/activate
uv pip install -e .

python -c "import compos3d_dp"
```

## Data Pipeline

### AWS & Anyscale Setup

#### Development

For local development, configure AWS and Anyscale access:

```bash
# Configure AWS SSO (uses your personal credentials)
aws configure sso

# Setup Anyscale (interactive login)
anyscale login
```

#### Production/Staging (First-Time)

For production or staging deployments, use Terraform and AWS Secrets Manager:

```bash
# Deploy infrastructure
cd terraform
terraform init
terraform apply -var-file=environments/prod.tfvars  # or staging.tfvars

# Store API keys in AWS Secrets Manager
aws secretsmanager put-secret-value \
  --secret-id compos3d-prod-anyscale-key \
  --secret-string "your-anyscale-api-key"

aws secretsmanager put-secret-value \
  --secret-id compos3d-prod-wandb-key \
  --secret-string "your-wandb-api-key"
```

> **Note:** Production/staging jobs use IAM roles and Secrets Manager (no local credentials needed).

### Deployment

#### Deploy Dev Environment

```bash
cd terraform
terraform init
terraform apply -var-file=environments/dev.tfvars -auto-approve
```

#### Deploy Staging Environment

```bash
cd terraform
terraform workspace new staging  # or: terraform workspace select staging
terraform apply -var-file=environments/staging.tfvars -auto-approve
```

#### Deploy Prod Environment

```bash
cd terraform
terraform workspace new prod  # or: terraform workspace select prod
terraform apply -var-file=environments/prod.tfvars -auto-approve
```

This creates the following resource per environment:

- S3 Buckets: Bronze, Silver, Gold (data lake layers)
- IAM Roles: Pipeline execution roles with proper permissions
- Secrets Manager: API key storage
- Glue Data Catalog: Metadata management
- Athena Workgroup: Query interface

### Running Pipelines

The data pipeline consists of three stages:
1. **Bronze Ingestion**: Ingest raw 3D scenes from BlenderBench dataset
2. **Silver Transformation**: Clean, validate, and transform data
3. **Gold Aggregation**: Create training-ready datasets

#### Run Complete Pipeline Locally

Test the complete pipeline on your local machine:

```bash
# Activate virtual environment
source .venv/bin/activate

# Run pipeline with 3 scenes (for testing)
python scripts/run_pipeline_on_anyscale.py --env dev --num-scenes 3 --local
```

#### Run Pipelines on Anyscale

Submit the pipeline job to Anyscale cloud:

```bash
source .venv/bin/activate

python submit_job.py
```

To customize the pipeline (edit `anyscale_job.yaml`):
- Change `--num-scenes 10` to process more scenes
- Change `--env dev` to `staging` or `prod`

```yaml
entrypoint: |
  pip install -e . && python scripts/run_pipeline_on_anyscale.py --env dev --num-scenes 10 --local
```

**Note**: The job automatically makes the container

Pipeline Output:
- Bronze layer: `s3://compos3d-{env}-bronze/compos3d/bronze/scenes/YYYY/MM/DD/`
- Silver layer: `s3://compos3d-{env}-silver/compos3d/silver/scenes/YYYY/MM/DD/`
- Gold layer: `s3://compos3d-{env}-gold/compos3d/gold/training_datasets/YYYY/MM/DD/`

#### Verify Pipeline Results

Check data was written to S3:

```bash
# Check Bronze layer
aws s3 ls s3://compos3d-dev-bronze/compos3d/bronze/scenes/$(date +%Y/%m/%d)/ --recursive

# Check Silver layer  
aws s3 ls s3://compos3d-dev-silver/compos3d/silver/scenes/$(date +%Y/%m/%d)/ --recursive

# Download and view Gold training dataset
aws s3 cp s3://compos3d-dev-gold/compos3d/gold/training_datasets/$(date +%Y/%m/%d)/dataset.json - | python -m json.tool
```

### Run Tests

Run the tests:

```bash
source .venv/bin/activate

# Run all tests
pytest tests/ -v

pytest tests/ -m bronze      # Bronze ingestion tests
pytest tests/ -m silver      # Silver transformation tests
pytest tests/ -m gold        # Gold aggregation tests
pytest tests/ -m training    # Training pipeline tests
pytest tests/ -m generation  # Generation pipeline tests

pytest tests/ -m unit
pytest tests/ -m integration
```