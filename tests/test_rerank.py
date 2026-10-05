from __future__ import annotations

import httpx

from stacmem.rerank import DashScopeReranker


class FakeClient:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def post(self, url: str, *, json: dict) -> httpx.Response:
        self.calls.append({"url": url, "json": json})
        documents = json["input"]["documents"]
        results = [
            {"index": index, "relevance_score": float(index + 1) / 10}
            for index in reversed(range(len(documents)))
        ]
        return httpx.Response(
            200,
            json={"output": {"results": results}},
            request=httpx.Request("POST", url),
        )

    def close(self) -> None:
        pass


def test_dashscope_rerank_restores_input_order_across_batches() -> None:
    reranker = DashScopeReranker(api_key="test", batch_size=2)
    reranker.client.close()
    fake = FakeClient()
    reranker.client = fake  # type: ignore[assignment]
    try:
        scores = reranker.score("query", ["a", "b", "c"])
    finally:
        reranker.close()
    assert scores == [0.1, 0.2, 0.1]
    assert len(fake.calls) == 2
