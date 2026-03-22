"""EC2 spot/on-demand job launcher for compos3d training and inference.

Design
------
The runner launches a single EC2 instance whose **user-data** script
bootstraps the environment and runs a ``compos3d`` CLI command.  All outputs
are written directly to S3 (bronze/silver/gold buckets) by the job itself via
the standard ``--env`` flag mechanism.  CloudWatch Logs capture stdout/stderr.
The instance self-terminates when the job finishes.

No SSH keys are required — interaction is via SSM (Systems Manager), so the
instance only needs the ``compos3d-ec2-job`` IAM instance profile with
SSM and S3 permissions.

Typical usage
-------------
    from compos3d.app_config import load_app_config
    from compos3d.compute.ec2_runner import EC2JobRunner, EC2JobSpec

    cfg = load_app_config("dev")
    runner = EC2JobRunner(app_config=cfg)

    spec = EC2JobSpec(
        command="train-hypotheses",
        cli_args=[
            "--dataset-path", "s3://compos3d-dev-bronze/datasets/vertical_slice.json",
            "--output-dir", "/tmp/training_out",
            "--config-path", "configs/compos3d.json",
            "--env", "dev",
        ],
        repo_url="https://github.com/yourorg/Compos3D.git",
        git_ref="main",
        log_group="/compos3d/jobs",
    )

    instance_id, job_info = runner.launch(spec)
    print(f"Job running on {instance_id}")
    runner.wait(instance_id, poll_interval_seconds=30)
"""

from __future__ import annotations

import base64
import textwrap
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import boto3

from compos3d.app_config import AppConfig


@dataclass
class EC2JobSpec:
    """Specification for a single EC2 job."""

    command: str
    """The compos3d CLI sub-command, e.g. ``train-hypotheses`` or ``run-inference``."""

    cli_args: list[str]
    """Arguments to pass to the CLI command (after the sub-command)."""

    repo_url: str = "https://github.com/yourorg/Compos3D.git"
    """Git remote to clone the project from on the instance."""

    git_ref: str = "main"
    """Branch, tag, or commit SHA to check out."""

    python_version: str = "3.11"
    """Python version to install in the bootstrap environment."""

    log_group: str = "/compos3d/jobs"
    """CloudWatch Logs log group for job stdout/stderr."""

    extra_setup_commands: list[str] = field(default_factory=list)
    """Additional shell commands to run after ``pip install`` but before the job."""

    job_timeout_minutes: int = 360
    """Terminate the instance after this many minutes even if the job is still running."""


class EC2JobRunner:
    """Launches EC2 spot/on-demand instances that run compos3d CLI jobs."""

    def __init__(self, app_config: AppConfig) -> None:
        self.cfg = app_config
        self.ec2 = boto3.client("ec2", region_name=app_config.aws_region)
        self.ssm = boto3.client("ssm", region_name=app_config.aws_region)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def launch(self, spec: EC2JobSpec) -> tuple[str, dict[str, Any]]:
        """Launch an EC2 instance that runs *spec*.

        Returns
        -------
        instance_id
            The EC2 instance ID.
        job_info
            Dict with instance_id, log_group, log_stream, and other metadata.
        """
        ami_id = self._resolve_ami()
        user_data = self._build_user_data(spec)
        user_data_b64 = base64.b64encode(user_data.encode()).decode()

        launch_kwargs: dict[str, Any] = {
            "ImageId": ami_id,
            "InstanceType": self.cfg.ec2_instance_type,
            "MinCount": 1,
            "MaxCount": 1,
            "UserData": user_data_b64,
            "IamInstanceProfile": {"Name": self.cfg.ec2_iam_instance_profile},
            "TagSpecifications": [
                {
                    "ResourceType": "instance",
                    "Tags": [
                        {"Key": "Name", "Value": f"compos3d-job-{spec.command}"},
                        {"Key": "Project", "Value": "compos3d"},
                        {"Key": "Environment", "Value": self.cfg.env},
                        {"Key": "ManagedBy", "Value": "compos3d-ec2-runner"},
                    ],
                }
            ],
            "InstanceInitiatedShutdownBehavior": "terminate",
            "MetadataOptions": {
                "HttpTokens": "required",  # IMDSv2
                "HttpEndpoint": "enabled",
            },
        }

        if self.cfg.ec2_subnet_id:
            launch_kwargs["SubnetId"] = self.cfg.ec2_subnet_id
        if self.cfg.ec2_security_group_id:
            launch_kwargs["SecurityGroupIds"] = [self.cfg.ec2_security_group_id]
        if self.cfg.ec2_key_name:
            launch_kwargs["KeyName"] = self.cfg.ec2_key_name

        if self.cfg.ec2_spot:
            launch_kwargs["InstanceMarketOptions"] = {
                "MarketType": "spot",
                "SpotOptions": {
                    "SpotInstanceType": "one-time",
                    **(
                        {"MaxPrice": self.cfg.ec2_spot_max_price}
                        if self.cfg.ec2_spot_max_price
                        else {}
                    ),
                },
            }

        resp = self.ec2.run_instances(**launch_kwargs)
        instance_id: str = resp["Instances"][0]["InstanceId"]

        log_stream = f"{instance_id}/{spec.command}"
        print(f"[ec2_runner] Launched {instance_id} ({self.cfg.ec2_instance_type})")
        print(f"[ec2_runner] CloudWatch log stream: {spec.log_group}/{log_stream}")
        print(f"[ec2_runner] Monitor: aws logs tail {spec.log_group} --log-stream-names {log_stream} --follow")

        return instance_id, {
            "instance_id": instance_id,
            "ami_id": ami_id,
            "instance_type": self.cfg.ec2_instance_type,
            "spot": self.cfg.ec2_spot,
            "log_group": spec.log_group,
            "log_stream": log_stream,
            "env": self.cfg.env,
            "command": spec.command,
        }

    def wait(
        self,
        instance_id: str,
        *,
        poll_interval_seconds: int = 30,
        timeout_minutes: int = 480,
    ) -> str:
        """Block until the instance terminates (job done) or timeout.

        Returns the final instance state (``"terminated"`` on success).
        """
        deadline = time.time() + timeout_minutes * 60
        print(f"[ec2_runner] Waiting for {instance_id} to terminate …")
        while time.time() < deadline:
            resp = self.ec2.describe_instances(InstanceIds=[instance_id])
            state = resp["Reservations"][0]["Instances"][0]["State"]["Name"]
            print(f"[ec2_runner] {instance_id} state: {state}")
            if state in ("terminated", "stopped"):
                return state
            time.sleep(poll_interval_seconds)
        raise TimeoutError(
            f"Instance {instance_id} did not terminate within {timeout_minutes} minutes."
        )

    def terminate(self, instance_id: str) -> None:
        """Force-terminate an instance."""
        self.ec2.terminate_instances(InstanceIds=[instance_id])
        print(f"[ec2_runner] Terminating {instance_id}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_ami(self) -> str:
        """Return the configured AMI, or look up the latest Deep Learning AMI."""
        if self.cfg.ec2_ami_id:
            return self.cfg.ec2_ami_id

        # Latest AWS Deep Learning AMI (Ubuntu 22.04, x86_64).
        resp = self.ec2.describe_images(
            Owners=["amazon"],
            Filters=[
                {"Name": "name", "Values": ["Deep Learning OSS Nvidia Driver AMI GPU PyTorch * (Ubuntu 22.04) *"]},
                {"Name": "state", "Values": ["available"]},
                {"Name": "architecture", "Values": ["x86_64"]},
            ],
        )
        images = sorted(resp["Images"], key=lambda img: img["CreationDate"], reverse=True)
        if not images:
            raise RuntimeError(
                "Could not find a suitable Deep Learning AMI.  "
                "Set ec2_ami_id in your AppConfig or config/env.<env>.yaml."
            )
        ami_id: str = images[0]["ImageId"]
        print(f"[ec2_runner] Using AMI {ami_id} ({images[0]['Name'][:60]}…)")
        return ami_id

    def _build_user_data(self, spec: EC2JobSpec) -> str:
        """Build the bash user-data script that bootstraps and runs the job."""
        cli_args_str = " ".join(f'"{a}"' if " " in a else a for a in spec.cli_args)
        extra_setup = "\n".join(spec.extra_setup_commands)

        # The script:
        #  1. Redirects all output to CloudWatch Logs via the CW agent.
        #  2. Clones the repo and installs dependencies.
        #  3. Runs the compos3d command with --env to route outputs to S3.
        #  4. Shuts the instance down on completion (triggering termination).
        script = textwrap.dedent(f"""\
            #!/bin/bash
            set -euxo pipefail

            LOG_GROUP="{spec.log_group}"
            LOG_STREAM="{spec.command}-$(ec2-metadata --instance-id | cut -d' ' -f2)"
            REGION="{self.cfg.aws_region}"

            # Install CloudWatch Logs agent if not already present
            if ! command -v amazon-cloudwatch-agent-ctl &> /dev/null; then
                apt-get install -y amazon-cloudwatch-agent 2>/dev/null || true
            fi

            # Create CW log group (idempotent)
            aws logs create-log-group --log-group-name "$LOG_GROUP" --region "$REGION" 2>/dev/null || true
            aws logs create-log-stream \\
                --log-group-name "$LOG_GROUP" \\
                --log-stream-name "$LOG_STREAM" \\
                --region "$REGION" 2>/dev/null || true

            # Helper: push a line to CloudWatch
            cw_log() {{
                TIMESTAMP=$(date +%s%3N)
                aws logs put-log-events \\
                    --log-group-name "$LOG_GROUP" \\
                    --log-stream-name "$LOG_STREAM" \\
                    --log-events timestamp=$TIMESTAMP,message="$1" \\
                    --region "$REGION" 2>/dev/null || true
            }}

            cw_log "=== compos3d job starting: {spec.command} ==="

            # Install Blender headless (bpy 4.2)
            pip install --quiet bpy==4.2.0 2>&1 | tail -3

            # Clone repo
            cd /opt
            git clone --depth 1 --branch "{spec.git_ref}" "{spec.repo_url}" compos3d
            cd compos3d
            git submodule update --init infinigen

            # Install compos3d
            pip install --quiet -e . 2>&1 | tail -3

            {extra_setup}

            # Install imageio-ffmpeg
            pip install --quiet "imageio[ffmpeg]" 2>&1 | tail -3

            cw_log "=== dependencies ready, running job ==="

            # Run the job; outputs go to S3 via --env flag
            set +e
            compos3d {spec.command} {cli_args_str}
            EXIT_CODE=$?
            set -e

            cw_log "=== job finished with exit code $EXIT_CODE ==="

            # Terminate the instance
            shutdown -h now
        """)
        return script
