#!/usr/bin/env python3
"""Submit job to Anyscale programmatically"""

import os
import subprocess
import sys
import boto3
from botocore.exceptions import NoCredentialsError, ProfileNotFound

# Set environment variables
os.environ["ANYSCALE_API_KEY"] = (
    "aph0_CkgwRgIhAIH7xFfVKXoG5qH2lNV_Sv8RJXh9sw_N0bPjk_G5MXbFAiEAjf9t6BcesXnNbCP4oqo2hM3dkSMDgQEAo4Ep2jtkocUSYxIgmwJv70L0PTrLJgzNHOM86DNGYs6sfu-zPfHGpCO6UAYASIedXNyXzM3YXRlMndzd2UxZ2NoMTZ3aGNqMWx1anoyOgwI4uHjzQYQ6PvI4AJCDAi2xcXMBhDo-8jgAvIBAA"
)
profile_name = "myisb_IsbUsersPS-136268833180"

print("=== Submitting Compos3D Pipeline to Anyscale Cloud ===")
print()

# Get AWS credentials using boto3
print("Getting AWS credentials...")
try:
    session = boto3.Session(profile_name=profile_name)
    credentials = session.get_credentials()

    if credentials is None:
        print("❌ No AWS credentials found!")
        print(f"Please run: aws sso login --profile {profile_name}")
        sys.exit(1)

    frozen_creds = credentials.get_frozen_credentials()
    aws_access_key = frozen_creds.access_key
    aws_secret_key = frozen_creds.secret_key
    aws_session_token = frozen_creds.token

    print(f"  Access Key: {aws_access_key[:10]}... ✓")
    print("  Secret Key: ****** ✓")
    if aws_session_token:
        print(f"  Session Token: {aws_session_token[:20]}... ✓")
    print()

except (NoCredentialsError, ProfileNotFound) as e:
    print(f"❌ Error getting AWS credentials: {e}")
    print(f"Please run: aws sso login --profile {profile_name}")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error: {e}")
    print(
        f"Your AWS session may have expired. Please run: aws sso login --profile {profile_name}"
    )
    sys.exit(1)

# Submit to Anyscale
print("Submitting job to Anyscale...")
cmd = [
    "/home/rishit/Compos3D/.venv/bin/anyscale",
    "job",
    "submit",
    "-f",
    "anyscale_job.yaml",
    "--env",
    f"APP_AWS_ACCESS_KEY_ID={aws_access_key}",
    "--env",
    f"APP_AWS_SECRET_ACCESS_KEY={aws_secret_key}",
]

# Add session token if present (for SSO)
if aws_session_token:
    cmd.extend(["--env", f"APP_AWS_SESSION_TOKEN={aws_session_token}"])

print(f"Running: {' '.join(cmd[:4])} ...")
print()

result = subprocess.run(cmd, capture_output=True, text=True)

print(result.stdout)
if result.stderr:
    # Filter out starship errors
    stderr_lines = result.stderr.split("\n")
    meaningful_errors = [
        line for line in stderr_lines if "starship" not in line.lower()
    ]
    if meaningful_errors:
        print("STDERR:", "\n".join(meaningful_errors), file=sys.stderr)

if result.returncode == 0:
    print()
    print("✅ Job submitted to Anyscale!")
    print("Monitor at: https://console.anyscale.com/jobs")
else:
    print()
    print(f"❌ Job submission failed with exit code {result.returncode}")
    sys.exit(1)
