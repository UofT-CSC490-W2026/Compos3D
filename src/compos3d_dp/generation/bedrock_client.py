"""
Amazon Bedrock Client for LLM Code Generation

Uses Bedrock's Claude models for VIGA agent code generation.
"""

from __future__ import annotations
import boto3
import json
from typing import List, Dict
import base64


class BedrockClient:
    """
    Wrapper for Amazon Bedrock API.

    Supports Claude models (Anthropic) via Bedrock.
    """

    def __init__(
        self,
        api_key: str,
        region: str = "us-east-1",
        model_id: str = "us.anthropic.claude-3-5-sonnet-20240620-v1:0",
    ):
        """
        Initialize Bedrock client.

        Args:
            api_key: Bedrock API key (base64 encoded)
            region: AWS region
            model_id: Model identifier
        """
        # Decode API key
        self.api_key = api_key

        # Parse API key format: "BedrockAPIKey-{profile}:{credentials}"
        try:
            decoded = base64.b64decode(api_key).decode("utf-8")
            # Format: BedrockAPIKey-fxfm-at-136268833180:O4EWPJb9KkQn4ZCSHEmCqHIRl4QA0GpjECRLYx6YDGDt7WTPUQzoRidesNw=
            parts = decoded.split(":", 1)
            if len(parts) == 2:
                parts[0].replace("BedrockAPIKey-", "")
                parts[1]
            else:
                raise ValueError("Invalid API key format")
        except Exception as e:
            print(f"⚠️  Could not parse API key: {e}")
            # Try to use it directly

        # Initialize Bedrock client
        # For now, we'll use the default AWS credentials from environment
        self.client = boto3.client(
            service_name="bedrock-runtime",
            region_name=region,
        )

        self.model_id = model_id

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> str:
        """
        Generate chat completion using Bedrock Claude.

        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            Generated text
        """
        # Convert messages to Claude format
        system_prompt = ""
        claude_messages = []

        for msg in messages:
            if msg["role"] == "system":
                system_prompt = msg["content"]
            else:
                claude_messages.append({"role": msg["role"], "content": msg["content"]})

        # Build request
        request_body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": claude_messages,
        }

        if system_prompt:
            request_body["system"] = system_prompt

        # Call Bedrock
        try:
            response = self.client.invoke_model(
                modelId=self.model_id,
                body=json.dumps(request_body),
            )

            # Parse response
            response_body = json.loads(response["body"].read())

            # Extract text
            content = response_body.get("content", [])
            if content and len(content) > 0:
                return content[0].get("text", "")

            return ""

        except Exception:
            raise


if __name__ == "__main__":
    # Test Bedrock client
    api_key = "ABSKQmVkcm9ja0FQSUtleS1meGZtLWF0LTEzNjI2ODgzMzE4MDpPNEVXUEpiOUtrUW40WkNTSEVtQ3FISVJsNFFBMEdwakVDUkxZeDZZREdEdDdXVFBVUXpvUmlkZXNudz0="

    print("🧪 Testing Bedrock client...")

    client = BedrockClient(api_key=api_key)

    # Simple test
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Say 'Hello, Bedrock!' and nothing else."},
    ]

    response = client.chat_completion(messages, temperature=0.0, max_tokens=100)
