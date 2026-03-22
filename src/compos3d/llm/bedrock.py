from dataclasses import dataclass
from typing import Any, Mapping, Sequence
import os

import boto3


@dataclass(frozen=True)
class BedrockConfig:
    model_id: str = "qwen.qwen3-next-80b-a3b"
    region_name: str = "us-east-1"

    @classmethod
    def from_env(cls) -> "BedrockConfig":
        return cls(
            model_id=os.environ.get("COMPOS3D_BEDROCK_MODEL_ID", cls.model_id),
            region_name=os.environ.get("COMPOS3D_BEDROCK_REGION", cls.region_name),
        )


def resolve_bedrock_config(
    *, model_id: str | None = None, region_name: str | None = None
) -> BedrockConfig:
    defaults = BedrockConfig.from_env()
    return BedrockConfig(
        model_id=model_id or defaults.model_id,
        region_name=region_name or defaults.region_name,
    )


class BedrockChatClient:
    def __init__(
        self, config: BedrockConfig | None = None, *, runtime_client: Any | None = None
    ) -> None:
        self.config = config or BedrockConfig.from_env()
        self.runtime_client = runtime_client or boto3.client(
            "bedrock-runtime",
            region_name=self.config.region_name,
        )

    def converse(
        self,
        *,
        messages: Sequence[Mapping[str, Any]],
        system: Sequence[Mapping[str, Any]] | None = None,
        inference_config: Mapping[str, Any] | None = None,
        additional_model_request_fields: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        request: dict[str, Any] = {
            "modelId": self.config.model_id,
            "messages": list(messages),
        }
        if system:
            request["system"] = list(system)
        if inference_config:
            request["inferenceConfig"] = dict(inference_config)
        if additional_model_request_fields:
            request["additionalModelRequestFields"] = dict(
                additional_model_request_fields
            )
        return self.runtime_client.converse(**request)
