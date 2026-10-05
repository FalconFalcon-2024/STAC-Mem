"""Embedding providers with a deterministic offline implementation."""

from __future__ import annotations

import hashlib
import math
import os
import re
from collections.abc import Sequence
from itertools import pairwise
from typing import Protocol

import numpy as np
from openai import OpenAI

_TOKEN_RE = re.compile(r"[\w\-]+", re.UNICODE)


class Embedder(Protocol):
    dimensions: int

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class HashingEmbedder:
    """Stable feature hashing for tests and API-free ablations."""

    def __init__(self, dimensions: int = 1024) -> None:
        self.dimensions = dimensions

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._one(text) for text in texts]

    def _one(self, text: str) -> list[float]:
        vector = np.zeros(self.dimensions, dtype=np.float32)
        tokens = _TOKEN_RE.findall(text.casefold())
        features = [*tokens, *(f"{a}::{b}" for a, b in pairwise(tokens))]
        for feature in features:
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
            raw = int.from_bytes(digest, "little")
            index = raw % self.dimensions
            sign = 1.0 if raw & 1 else -1.0
            vector[index] += sign
        norm = float(np.linalg.norm(vector))
        if norm > 0:
            vector /= norm
        return vector.tolist()


class OpenAICompatibleEmbedder:
    def __init__(
        self,
        *,
        model: str = "text-embedding-v4",
        dimensions: int = 1024,
        api_key: str | None = None,
        base_url: str | None = None,
        batch_size: int = 10,
        api_key_env: str = "OPENAI_API_KEY",
        timeout_seconds: float = 120.0,
        max_retries: int = 4,
        send_dimensions: bool = True,
    ) -> None:
        key = api_key or os.getenv(api_key_env)
        if not key:
            raise ValueError(f"Set {api_key_env} for the configured embedding provider")
        if not base_url:
            raise ValueError("An explicit embedding base_url is required")
        self.model = model
        self.dimensions = dimensions
        self.batch_size = batch_size
        self.send_dimensions = send_dimensions
        self.client = OpenAI(
            api_key=key,
            base_url=base_url,
            timeout=timeout_seconds,
            max_retries=max_retries,
        )

    def close(self) -> None:
        self.client.close()

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        output: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start : start + self.batch_size])
            if not batch:
                continue
            response = self.client.embeddings.create(
                model=self.model,
                input=batch,
                **({"dimensions": self.dimensions} if self.send_dimensions else {}),
            )
            items = sorted(response.data, key=lambda item: item.index)
            if [item.index for item in items] != list(range(len(batch))):
                raise ValueError("Embedding response indices do not match the input batch")
            for item in items:
                if len(item.embedding) != self.dimensions or not all(
                    math.isfinite(value) for value in item.embedding
                ):
                    raise ValueError(
                        "Embedding response has invalid dimensions or nonfinite values"
                    )
                output.append(item.embedding)
        return output


class QwenEmbedder(OpenAICompatibleEmbedder):
    """Backward-compatible provider preset."""

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("api_key_env", "DASHSCOPE_API_KEY")
        kwargs["base_url"] = kwargs.get("base_url") or os.getenv(
            "STACMEM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        super().__init__(**kwargs)


def cosine_similarity(a: Sequence[float] | None, b: Sequence[float] | None) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    left = np.asarray(a, dtype=np.float32)
    right = np.asarray(b, dtype=np.float32)
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denom <= 0 or not math.isfinite(denom):
        return 0.0
    return max(-1.0, min(1.0, float(np.dot(left, right) / denom)))
