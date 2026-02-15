import boto3
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)


def submit_batch_job(
    job_name: str,
    job_queue: str,
    job_definition: str,
    command_overrides: List[str],
    environment_overrides: Dict[str, str] = None,
) -> str:
    """Submit a job to AWS Batch."""
    batch = boto3.client("batch")

    container_overrides = {"command": command_overrides}

    if environment_overrides:
        container_overrides["environment"] = [
            {"name": k, "value": v} for k, v in environment_overrides.items()
        ]

    try:
        response = batch.submit_job(
            jobName=job_name,
            jobQueue=job_queue,
            jobDefinition=job_definition,
            containerOverrides=container_overrides,
        )
        job_id = response["jobId"]
        logger.info(f"Submitted Batch job {job_name}: {job_id}")
        return job_id
    except Exception as e:
        logger.error(f"Failed to submit Batch job: {e}")
        raise


def trigger_silver_build(
    job_queue_arn: str, job_definition_arn: str, date_partition: str, env: str = "prod"
):
    """Trigger the silver-build pipeline on Batch."""
    command = [
        "python",
        "-m",
        "compos3d_dp.cli",
        "silver-build",
        "--env",
        env,
        "--date",
        date_partition,
    ]

    return submit_batch_job(
        job_name=f"silver-build-{date_partition}",
        job_queue=job_queue_arn,
        job_definition=job_definition_arn,
        command_overrides=command,
    )
