"""
Compos3D - Sample Modal app: upload a dummy file to the Data Lake S3 bucket.

Prerequisites:
  1. Apply Terraform in terraform/environments/dev/ and note the S3 bucket name.
  2. Create an IAM user with the same S3 read/write policy as the Modal role
     (or use the role's policy attached to a user). Create access keys for that user.
  3. In Modal dashboard: Secrets -> create a secret named "aws-compos3d" with:
     - AWS_ACCESS_KEY_ID
     - AWS_SECRET_ACCESS_KEY
     - AWS_REGION (e.g. us-east-1)
  4. Optionally set BUCKET_NAME in the secret, or pass it when calling the function.

Run: modal run modal_app.py
"""

import modal

app = modal.App("compos3d-s3-upload")

# Use the secret that holds AWS credentials (populated from IAM user keys with S3 access)
@app.function(secrets=[modal.Secret.from_name("aws-compos3d")])
def upload_dummy_file_to_s3(bucket_name: str | None = None):
    """
    Upload a small dummy file to the Compos3D Data Lake S3 bucket.
    If bucket_name is not provided, reads BUCKET_NAME from the Modal secret.
    """
    import os
    import boto3
    from datetime import datetime

    bucket = bucket_name or os.environ.get("BUCKET_NAME")
    if not bucket:
        raise ValueError(
            "Provide bucket_name or set BUCKET_NAME in the 'aws-compos3d' Modal secret. "
            "Bucket name is in Terraform output: data_lake_bucket_name (e.g. dev-compos3d-data-lake)."
        )

    # boto3 will use AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY from the Modal secret
    client = boto3.client("s3", region_name=os.environ.get("AWS_REGION", "us-east-1"))

    key = "uploads/dummy.txt"
    body = f"Compos3D dummy upload at {datetime.utcnow().isoformat()}Z\n"

    client.put_object(Bucket=bucket, Key=key, Body=body, ContentType="text/plain")
    return {"bucket": bucket, "key": key, "message": "Upload successful"}


@app.local_entrypoint()
def main(bucket_name: str = ""):
    result = upload_dummy_file_to_s3.remote(bucket_name or None)
    print("Upload result:", result)
