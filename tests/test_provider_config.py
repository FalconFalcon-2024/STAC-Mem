from types import SimpleNamespace

import httpx
import pytest
from openai import OpenAI

from stacmem.config import AppConfig
from stacmem.embeddings import OpenAICompatibleEmbedder
from stacmem.pipeline import StacMemory
from stacmem.providers import OpenAIJsonClient, build_json_backend


def config(tmp_path):
    result = AppConfig()
    result.runtime.database_path = str(tmp_path / "memory.db")
    result.runtime.request_timeout_seconds = 17
    result.extraction.provider = "openai_compatible"
    result.extraction.base_url = "http://chat.invalid/v1"
    result.extraction.api_key_env = "TEST_CHAT_KEY"
    result.embedding.provider = "openai_compatible"
    result.embedding.base_url = "http://embed.invalid/v1"
    result.embedding.api_key_env = "TEST_EMBED_KEY"
    result.embedding.max_retries = 0
    result.rerank.provider = "none"
    return result


def test_independent_endpoints_timeout_and_close(tmp_path, monkeypatch):
    clients = []

    def client(**kwargs):
        instance = SimpleNamespace(options=kwargs, closed=False)
        instance.close = lambda: setattr(instance, "closed", True)
        clients.append(instance)
        return instance

    monkeypatch.setenv("TEST_CHAT_KEY", "chat-test")
    monkeypatch.setenv("TEST_EMBED_KEY", "embed-test")
    monkeypatch.setattr("stacmem.providers.OpenAI", client)
    monkeypatch.setattr("stacmem.embeddings.OpenAI", client)
    memory = StacMemory.from_app_config(config(tmp_path))
    assert len(clients) == 3
    assert clients[0].options["base_url"] == "http://embed.invalid/v1"
    assert clients[0].options["max_retries"] == 0
    assert clients[1].options["api_key"] == "chat-test"
    assert all(c.options["timeout"] == 17 for c in clients)
    memory.close()
    assert all(c.closed for c in clients)


def test_generic_endpoint_never_silently_defaults(tmp_path, monkeypatch):
    monkeypatch.delenv("STACMEM_BASE_URL", raising=False)
    cfg = config(tmp_path)
    cfg.extraction.base_url = None
    with pytest.raises(ValueError, match="requires base_url"):
        build_json_backend(cfg)


def test_missing_key_message_names_environment_not_secret(tmp_path, monkeypatch):
    monkeypatch.delenv("TEST_CHAT_KEY", raising=False)
    with pytest.raises(ValueError, match="TEST_CHAT_KEY"):
        build_json_backend(config(tmp_path))


def test_qwen_preset_remains_supported(tmp_path, monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")
    monkeypatch.delenv("STACMEM_BASE_URL", raising=False)
    captured = {}
    monkeypatch.setattr("stacmem.providers.OpenAI", lambda **kw: captured.update(kw))
    cfg = AppConfig()
    build_json_backend(cfg)
    assert captured["base_url"] == "https://dashscope.aliyuncs.com/compatible-mode/v1"
    assert captured["timeout"] == cfg.runtime.request_timeout_seconds


@pytest.mark.parametrize("json_mode", [False, True])
def test_json_mode_and_invalid_object(json_mode):
    backend = object.__new__(OpenAIJsonClient)
    backend.model, backend.temperature, backend.json_mode = "test", 0, json_mode
    options = {}

    def create(**kwargs):
        options.update(kwargs)
        return SimpleNamespace(
            usage=None, choices=[SimpleNamespace(message=SimpleNamespace(content="[]"))]
        )

    backend.client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    with pytest.raises(ValueError, match="must return an object"):
        backend.call("system", "user")
    assert ("response_format" in options) is json_mode
    assert backend.last_response_text == "[]"
    assert backend.last_usage["total_tokens"] == 0


def embedder(items):
    instance = object.__new__(OpenAICompatibleEmbedder)
    instance.model, instance.dimensions, instance.batch_size = "test", 2, 10
    instance.send_dimensions = False
    captured = {}

    def create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(data=[SimpleNamespace(index=i, embedding=v) for i, v in items])

    instance.client = SimpleNamespace(embeddings=SimpleNamespace(create=create))
    return instance, captured


def test_embedding_order_and_optional_dimensions():
    instance, captured = embedder([(1, [0, 1]), (0, [1, 0])])
    assert instance.embed(["a", "b"]) == [[1, 0], [0, 1]]
    assert "dimensions" not in captured


@pytest.mark.parametrize("items", [
    [(0, [1, 0])], [(0, [1, 0]), (0, [0, 1])],
    [(0, [1]), (1, [0, 1])], [(0, [float("nan"), 0]), (1, [0, 1])],
])
def test_invalid_embedding_response_is_rejected(items):
    instance, _ = embedder(items)
    with pytest.raises(ValueError, match="Embedding response"):
        instance.embed(["a", "b"])


def test_real_sdk_with_mock_transport(monkeypatch):
    import json

    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(200, json={
            "id": "reply", "object": "chat.completion", "created": 1, "model": "test",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": '{"ok":true}'},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
        })

    client = OpenAI(
        api_key="test-key", base_url="http://mock.invalid/v1", max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(respond)),
    )
    monkeypatch.setattr("stacmem.providers.OpenAI", lambda **kwargs: client)
    backend = OpenAIJsonClient(model="test", base_url="http://mock.invalid/v1", api_key="test-key")
    try:
        assert backend.call("json only", "hello") == {"ok": True}
        assert requests[0].url.path == "/v1/chat/completions"
        assert json.loads(requests[0].content)["response_format"] == {"type": "json_object"}
        assert backend.last_usage["total_tokens"] == 5
    finally:
        backend.close()
    assert client.is_closed()
