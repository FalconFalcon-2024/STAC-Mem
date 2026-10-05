"""OpenAI-compatible JSON transport, independent of memory semantics."""

from __future__ import annotations

import json
import os
from typing import Any, Protocol

from openai import OpenAI

from .config import AppConfig


class JsonBackend(Protocol):
    model: str
    last_usage: dict[str, int]

    def call(self, system: str, user: str) -> dict[str, Any]: ...


def endpoint_options(provider: str, base_url: str | None, api_key_env: str | None) -> dict:
    if provider not in {"qwen", "openai_compatible"}:
        raise ValueError(f"Unsupported online provider: {provider}")
    qwen = provider == "qwen"
    # A generic endpoint must be explicit; never silently send data to another provider.
    endpoint = base_url or os.getenv("STACMEM_BASE_URL")
    if not endpoint and qwen:
        endpoint = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    if not endpoint:
        raise ValueError("openai_compatible requires base_url or STACMEM_BASE_URL")
    return {
        "base_url": endpoint,
        "api_key_env": api_key_env or ("DASHSCOPE_API_KEY" if qwen else "OPENAI_API_KEY"),
    }


class OpenAIJsonClient:
    def __init__(
        self, *, model: str, base_url: str, api_key_env: str = "OPENAI_API_KEY",
        api_key: str | None = None, temperature: float = 0.0,
        max_retries: int = 4, timeout_seconds: float = 180.0, json_mode: bool = True,
    ) -> None:
        key = api_key or os.getenv(api_key_env)
        if not key:
            raise ValueError(f"Set {api_key_env} for the configured JSON provider")
        self.model = model
        self.temperature = temperature
        self.json_mode = json_mode
        self.client = OpenAI(
            api_key=key, base_url=base_url, timeout=timeout_seconds, max_retries=max_retries
        )
        self.last_usage: dict[str, int] = {}
        self.last_response_text: str | None = None

    def close(self) -> None:
        self.client.close()

    def call(self, system: str, user: str) -> dict[str, Any]:
        self.last_usage = {}
        self.last_response_text = None
        options = {"response_format": {"type": "json_object"}} if self.json_mode else {}
        response = self.client.chat.completions.create(
            model=self.model, temperature=self.temperature,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            **options,
        )
        usage = response.usage
        self.last_usage = {
            "prompt_tokens": int(usage.prompt_tokens if usage else 0),
            "completion_tokens": int(usage.completion_tokens if usage else 0),
            "total_tokens": int(usage.total_tokens if usage else 0),
        }
        content = response.choices[0].message.content or ""
        self.last_response_text = content
        stripped = content.strip()
        if stripped.startswith("```") and stripped.endswith("```"):
            stripped = "\n".join(stripped.splitlines()[1:-1])
        result = json.loads(stripped)
        if not isinstance(result, dict):
            raise ValueError("JSON provider must return an object")
        return result


def build_json_backend(config: AppConfig) -> OpenAIJsonClient:
    extraction = config.extraction
    return OpenAIJsonClient(
        model=extraction.model,
        **endpoint_options(extraction.provider, extraction.base_url, extraction.api_key_env),
        temperature=extraction.temperature,
        max_retries=extraction.max_retries,
        timeout_seconds=config.runtime.request_timeout_seconds,
        json_mode=extraction.json_mode,
    )
