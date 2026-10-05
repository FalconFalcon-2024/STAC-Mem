"""Rerank providers used before spatio-temporal and conflict scoring."""

from __future__ import annotations

import os
import time
from collections.abc import Sequence
from typing import Any, Protocol

import httpx


class Reranker(Protocol):
    def score(self, query: str, documents: Sequence[str]) -> list[float]: ...


class DashScopeReranker:
    _SERVICE_PATH = "/api/v1/services/rerank/text-rerank/text-rerank"

    def __init__(
        self,
        *,
        model: str = "gte-rerank-v2",
        api_key: str | None = None,
        base_url: str = "https://dashscope.aliyuncs.com",
        batch_size: int = 10,
        timeout_seconds: float = 60.0,
        max_retries: int = 3,
    ) -> None:
        key = api_key or os.getenv("DASHSCOPE_API_KEY")
        if not key:
            raise ValueError("DASHSCOPE_API_KEY is required for DashScope reranking")
        if model != "gte-rerank-v2":
            raise ValueError("the DashScope adapter currently supports gte-rerank-v2")
        self.model = model
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.url = f"{base_url.rstrip('/')}{self._SERVICE_PATH}"
        self.client = httpx.Client(
            timeout=httpx.Timeout(timeout_seconds, connect=10.0),
            headers={"Authorization": f"Bearer {key}"},
        )

    def close(self) -> None:
        self.client.close()

    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        scores = [0.0] * len(documents)
        for offset in range(0, len(documents), self.batch_size):
            batch = list(documents[offset : offset + self.batch_size])
            for index, score in self._score_batch(query, batch):
                scores[offset + index] = score
        return scores

    def _score_batch(self, query: str, documents: list[str]) -> list[tuple[int, float]]:
        payload: dict[str, Any] = {
            "model": self.model,
            "input": {"query": query, "documents": documents},
            "parameters": {"return_documents": False, "top_n": len(documents)},
        }
        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.post(self.url, json=payload)
            except httpx.HTTPError:
                if attempt == self.max_retries:
                    raise
                time.sleep(min(2**attempt, 8))
                continue
            if response.status_code == 200:
                body = response.json()
                output = body.get("output") if isinstance(body, dict) else None
                results = output.get("results") if isinstance(output, dict) else None
                if not isinstance(results, list):
                    raise RuntimeError(f"DashScope rerank response has no results: {body}")
                return [(int(item["index"]), float(item["relevance_score"])) for item in results]
            if response.status_code not in {429, 500, 502, 503, 504}:
                raise RuntimeError(
                    f"DashScope rerank HTTP {response.status_code}: {response.text[:300]}"
                )
            if attempt == self.max_retries:
                raise RuntimeError(
                    f"DashScope rerank exhausted retries: HTTP {response.status_code}"
                )
            time.sleep(min(2**attempt, 8))
        raise RuntimeError("unreachable rerank retry state")
