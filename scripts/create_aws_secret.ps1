# Creates Modal secret "aws-compos3d" from environment variables.
# Secrets never live in this repo: use a local .env file (gitignored) or set env vars.
#
# Setup:
#   1. cp .env.example .env   (or copy and rename)
#   2. Edit .env with real AWS credentials (IAM user access keys from AWS Console)
#   3. Run this script from repo root:  .\scripts\create_aws_secret.ps1

$ErrorActionPreference = "Stop"
$repoRoot = (Get-Item $PSScriptRoot).Parent.FullName
$envFile = Join-Path $repoRoot ".env"

if (Test-Path $envFile) {
    Get-Content $envFile | ForEach-Object {
        if ($_ -match '^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)\s*$' -and $_.Trim() -notmatch '^\#') {
            $key = $matches[1]
            $val = $matches[2].Trim()
            if ($val -match '^["''](.*)["'']$') { $val = $matches[1] }
            [Environment]::SetEnvironmentVariable($key, $val, "Process")
        }
    }
}

$required = @("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY")
foreach ($k in $required) {
    $v = [Environment]::GetEnvironmentVariable($k, "Process")
    if (-not $v -or $v -eq "") {
        Write-Host "Missing $k. Create .env from .env.example and fill in credentials, or set env vars." -ForegroundColor Red
        Write-Host "  .env is gitignored and must never be committed." -ForegroundColor Yellow
        exit 1
    }
}

$region = if ($env:AWS_REGION) { $env:AWS_REGION } else { "us-east-1" }
$bucket = if ($env:BUCKET_NAME) { $env:BUCKET_NAME } else { "dev-compos3d-data-lake" }

Write-Host "Creating Modal secret 'aws-compos3d' from .env / environment (no secrets in repo)." -ForegroundColor Green
modal secret create aws-compos3d `
  "AWS_ACCESS_KEY_ID=$env:AWS_ACCESS_KEY_ID" `
  "AWS_SECRET_ACCESS_KEY=$env:AWS_SECRET_ACCESS_KEY" `
  "AWS_REGION=$region" `
  "BUCKET_NAME=$bucket"
