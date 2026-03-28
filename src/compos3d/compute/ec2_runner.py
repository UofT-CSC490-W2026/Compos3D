"""EC2 spot/on-demand job launcher for compos3d training and inference.

The launcher keeps local workflows untouched and adds an AWS-native execution
path around the existing CLI:

- pull a prebuilt Compos3D runtime image from ECR
- run a single container per job on an ephemeral EC2 instance
- stream logs to CloudWatch via Docker awslogs
- let the container-side runtime wrapper fetch secrets, hydrate S3 inputs,
  manage checkpoint sync, and then invoke the normal ``compos3d`` command
"""

from __future__ import annotations

import base64
import shlex
import textwrap
import time
from dataclasses import dataclass, field
from typing import Any

import boto3

from compos3d.app_config import AppConfig

_DLAMI_PARAMETER = (
    "/aws/service/deeplearning/ami/x86_64/"
    "base-oss-nvidia-driver-gpu-ubuntu-22.04/latest/ami-id"
)


@dataclass
class EC2JobSpec:
    """Specification for a single EC2 job."""

    command: str
    cli_args: list[str]
    repo_url: str = "https://github.com/yourorg/Compos3D.git"
    git_ref: str = "main"
    python_version: str = "3.11"
    log_group: str = "/compos3d/jobs"
    extra_setup_commands: list[str] = field(default_factory=list)
    job_timeout_minutes: int = 360
    image_tag: str = "latest"


class EC2JobRunner:
    """Launches EC2 spot/on-demand instances that run compos3d container jobs."""

    def __init__(self, app_config: AppConfig) -> None:
        self.cfg = app_config
        self.ec2 = boto3.client("ec2", region_name=app_config.aws_region)
        self.ssm = boto3.client("ssm", region_name=app_config.aws_region)
        self.ecr = boto3.client("ecr", region_name=app_config.aws_region)

    def launch(self, spec: EC2JobSpec) -> tuple[str, dict[str, Any]]:
        ami_id = self._resolve_ami()
        image_url = self._resolve_ecr_repository_url()
        image_ref = f"{image_url}:{spec.image_tag}"
        user_data = self._build_user_data(spec=spec, image_ref=image_ref)
        user_data_b64 = base64.b64encode(user_data.encode()).decode()
        subnet_id = self.cfg.ec2_subnet_id or self._discover_subnet_id()
        security_group_id = (
            self.cfg.ec2_security_group_id or self._discover_security_group_id()
        )

        launch_kwargs: dict[str, Any] = {
            "ImageId": ami_id,
            "InstanceType": self.cfg.ec2_instance_type,
            "MinCount": 1,
            "MaxCount": 1,
            "UserData": user_data_b64,
            "IamInstanceProfile": {"Name": self.cfg.ec2_iam_instance_profile},
            "BlockDeviceMappings": [
                {
                    "DeviceName": "/dev/sda1",
                    "Ebs": {
                        "VolumeSize": self.cfg.ec2_root_volume_size_gb,
                        "VolumeType": "gp3",
                        "DeleteOnTermination": True,
                    },
                }
            ],
            "TagSpecifications": [
                {
                    "ResourceType": "instance",
                    "Tags": [
                        {"Key": "Name", "Value": f"compos3d-job-{spec.command}"},
                        {"Key": "Project", "Value": "compos3d"},
                        {"Key": "Environment", "Value": self.cfg.env},
                        {"Key": "ManagedBy", "Value": "compos3d-ec2-runner"},
                        {"Key": "ImageRef", "Value": image_ref},
                    ],
                }
            ],
            "InstanceInitiatedShutdownBehavior": "terminate",
            "MetadataOptions": {
                "HttpTokens": "required",
                "HttpEndpoint": "enabled",
            },
        }

        if subnet_id:
            launch_kwargs["SubnetId"] = subnet_id
        if security_group_id:
            launch_kwargs["SecurityGroupIds"] = [security_group_id]
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
        print(f"[ec2_runner] Image: {image_ref}")
        print(f"[ec2_runner] CloudWatch log stream: {spec.log_group}/{log_stream}")
        print(
            f"[ec2_runner] Monitor: aws logs tail {spec.log_group} --log-stream-names {log_stream} --follow"
        )

        return instance_id, {
            "instance_id": instance_id,
            "ami_id": ami_id,
            "instance_type": self.cfg.ec2_instance_type,
            "spot": self.cfg.ec2_spot,
            "log_group": spec.log_group,
            "log_stream": log_stream,
            "env": self.cfg.env,
            "command": spec.command,
            "image_ref": image_ref,
            "root_volume_size_gb": self.cfg.ec2_root_volume_size_gb,
            "subnet_id": subnet_id,
            "security_group_id": security_group_id,
            "repo_url": spec.repo_url,
            "git_ref": spec.git_ref,
        }

    def wait(
        self,
        instance_id: str,
        *,
        poll_interval_seconds: int = 30,
        timeout_minutes: int = 480,
    ) -> str:
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
        self.ec2.terminate_instances(InstanceIds=[instance_id])
        print(f"[ec2_runner] Terminating {instance_id}")

    def _resolve_ami(self) -> str:
        if self.cfg.ec2_ami_id:
            return self.cfg.ec2_ami_id

        try:
            response = self.ssm.get_parameter(Name=_DLAMI_PARAMETER)
            ami_id = response["Parameter"]["Value"]
            print(f"[ec2_runner] Using DLAMI from SSM parameter: {ami_id}")
            return ami_id
        except Exception:  # noqa: BLE001
            pass

        resp = self.ec2.describe_images(
            Owners=["amazon"],
            Filters=[
                {
                    "Name": "name",
                    "Values": [
                        "Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04) *",
                        "Deep Learning OSS Nvidia Driver AMI GPU PyTorch * (Ubuntu 22.04) *",
                    ],
                },
                {"Name": "state", "Values": ["available"]},
                {"Name": "architecture", "Values": ["x86_64"]},
            ],
        )
        images = sorted(
            resp["Images"], key=lambda img: img["CreationDate"], reverse=True
        )
        if not images:
            raise RuntimeError(
                "Could not find a suitable GPU Deep Learning AMI. "
                "Set ec2_ami_id in your AppConfig or config/env.<env>.yaml."
            )
        ami_id: str = images[0]["ImageId"]
        print(f"[ec2_runner] Using AMI {ami_id} ({images[0]['Name'][:60]}…)")
        return ami_id

    def _resolve_ecr_repository_url(self) -> str:
        if self.cfg.ecr_repository_url:
            return self.cfg.ecr_repository_url

        repository_name = f"compos3d-{self.cfg.env}-runtime"
        response = self.ecr.describe_repositories(repositoryNames=[repository_name])
        repositories = response.get("repositories", [])
        if not repositories:
            raise RuntimeError(
                "Could not resolve ECR repository URL. Set ecr_repository_url in "
                "config/env.<env>.yaml or create the expected repository via Terraform."
            )
        return repositories[0]["repositoryUri"]

    def _discover_subnet_id(self) -> str | None:
        response = self.ec2.describe_subnets(
            Filters=[
                {"Name": "tag:Project", "Values": ["compos3d"]},
                {"Name": "tag:Environment", "Values": [self.cfg.env]},
                {"Name": "tag:Tier", "Values": ["public"]},
                {"Name": "state", "Values": ["available"]},
            ]
        )
        subnets = sorted(
            response.get("Subnets", []),
            key=lambda subnet: (
                subnet.get("AvailabilityZone", ""),
                subnet.get("SubnetId", ""),
            ),
        )
        if not subnets:
            return None
        subnet_id = subnets[0]["SubnetId"]
        print(f"[ec2_runner] Using discovered subnet {subnet_id}")
        return subnet_id

    def _discover_security_group_id(self) -> str | None:
        response = self.ec2.describe_security_groups(
            Filters=[
                {"Name": "tag:Project", "Values": ["compos3d"]},
                {"Name": "tag:Environment", "Values": [self.cfg.env]},
                {"Name": "group-name", "Values": [f"compos3d-{self.cfg.env}-ec2-job"]},
            ]
        )
        groups = sorted(
            response.get("SecurityGroups", []),
            key=lambda group: group.get("GroupId", ""),
        )
        if not groups:
            return None
        group_id = groups[0]["GroupId"]
        print(f"[ec2_runner] Using discovered security group {group_id}")
        return group_id

    def _build_user_data(self, *, spec: EC2JobSpec, image_ref: str) -> str:
        extra_setup = "\n".join(spec.extra_setup_commands)
        gpu_flag = (
            "--gpus all" if self._requires_gpu(self.cfg.ec2_instance_type) else ""
        )
        cli_tokens = [
            "--runtime-env",
            str(self.cfg.env),
            "--instance-type",
            str(self.cfg.ec2_instance_type),
            spec.command,
            *spec.cli_args,
        ]
        container_cmd = shlex.join(cli_tokens)
        ecr_registry = image_ref.rsplit("/", 1)[0]

        script = textwrap.dedent(
            f"""\
            #!/bin/bash
            set -euxo pipefail
            export DEBIAN_FRONTEND=noninteractive

            REGION={shlex.quote(self.cfg.aws_region)}
            LOG_GROUP={shlex.quote(spec.log_group)}
            IMAGE_REF={shlex.quote(image_ref)}
            ECR_REGISTRY={shlex.quote(ecr_registry)}
            JOB_TIMEOUT_MINUTES={int(spec.job_timeout_minutes)}

            apt-get update -y
            apt-get install -y awscli curl jq
            if ! command -v docker >/dev/null 2>&1; then
              apt-get install -y docker.io
            fi
            systemctl enable --now docker || systemctl start docker

            if command -v nvidia-ctk >/dev/null 2>&1; then
              nvidia-ctk runtime configure --runtime=docker || true
              systemctl restart docker || true
            fi

            TOKEN=$(curl -fsS -X PUT "http://169.254.169.254/latest/api/token" \\
              -H "X-aws-ec2-metadata-token-ttl-seconds: 21600")
            INSTANCE_ID=$(curl -fsS -H "X-aws-ec2-metadata-token: $TOKEN" \\
              http://169.254.169.254/latest/meta-data/instance-id)
            LOG_STREAM="${{INSTANCE_ID}}/{spec.command}"

            aws ecr get-login-password --region "$REGION" | \\
              docker login --username AWS --password-stdin "$ECR_REGISTRY"

            mkdir -p /opt/compos3d/work /opt/compos3d/work/artifacts

            {extra_setup}

            docker pull "$IMAGE_REF"

            set +e
            timeout "${{JOB_TIMEOUT_MINUTES}}m" docker run --rm {gpu_flag} \\
              --log-driver=awslogs \\
              --log-opt awslogs-region="$REGION" \\
              --log-opt awslogs-group="$LOG_GROUP" \\
              --log-opt awslogs-stream="$LOG_STREAM" \\
              --log-opt awslogs-create-group=true \\
              -e AWS_DEFAULT_REGION="$REGION" \\
              -e COMPOS3D_ENV={shlex.quote(str(self.cfg.env))} \\
              -e COMPOS3D_INSTANCE_TYPE={shlex.quote(str(self.cfg.ec2_instance_type))} \\
              -v /opt/compos3d/work:/work \\
              -v /opt/compos3d/work/artifacts:/opt/compos3d/artifacts \\
              "$IMAGE_REF" {container_cmd}
            EXIT_CODE=$?
            set -e

            shutdown -h now
            exit $EXIT_CODE
            """
        )
        return script

    @staticmethod
    def _requires_gpu(instance_type: str | None) -> bool:
        if not instance_type:
            return False
        normalized = instance_type.lower()
        return normalized.startswith(("g", "p", "trn", "inf"))
