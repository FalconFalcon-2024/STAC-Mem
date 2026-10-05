from __future__ import annotations

from types import SimpleNamespace

import pytest

from stacmem.extractors.qwen import (
    QwenClaimExtractor,
    QwenQueryCompiler,
    _QwenJsonClient,
    parse_model_place,
    parse_model_time,
)
from stacmem.models import Message
from stacmem.time_utils import ensure_ms


def test_model_place_drops_null_list_members_and_unknown_keys() -> None:
    place = parse_model_place(
        {
            "place_id": None,
            "name": "Suzhou",
            "aliases": ["Soochow", None, ""],
            "hierarchy": ["CN", "Jiangsu", "Suzhou", None],
            "latitude": 31.2989,
            "longitude": 120.5853,
            "radius_km": None,
            "role": "scope",
            "unexpected": "ignored",
        }
    )

    assert place is not None
    assert place.aliases == ["Soochow"]
    assert place.hierarchy == ["CN", "Jiangsu", "Suzhou"]
    assert place.name == "Suzhou"


def test_model_place_accepts_null_or_missing_lists() -> None:
    place = parse_model_place({"name": "Chengdu", "aliases": None, "hierarchy": None})

    assert place is not None
    assert place.aliases == []
    assert place.hierarchy == []


def test_nullable_model_time_preserves_unknown_and_rejects_invalid_dates() -> None:
    for value in (None, "null", " NULL ", ""):
        assert parse_model_time(value) is None
    assert parse_model_time("2027-02-03T10:00:00+08:00") == ensure_ms("2027-02-03T02:00:00Z")
    for value in ("unknown", "2027-02-30", True):
        with pytest.raises(ValueError):
            parse_model_time(value)


def test_extractor_records_quoted_null_without_changing_valid_time_semantics() -> None:
    extractor = object.__new__(QwenClaimExtractor)
    payload = {
        "claims": [
            {
                "subject": "user",
                "predicate": "employment.organization",
                "object_value": "Company A",
                "valid_start": "null",
                "valid_end": "2027-04-05",
                "source_content": "I used to work at Company A.",
            }
        ]
    }
    extractor.backend = SimpleNamespace(
        call=lambda *_: payload,
        model="fake",
        last_usage={"total_tokens": 10},
    )
    claims = extractor.extract(
        owner_id="user",
        session_id="s1",
        messages=[
            Message(
                sender_id="user",
                role="user",
                timestamp=ensure_ms("2027-05-01"),
                content="I used to work at Company A.",
            )
        ],
    )
    assert len(claims) == 1
    assert claims[0].valid_start is None
    assert claims[0].valid_end == ensure_ms("2027-04-05")
    assert claims[0].metadata["model_null_time_normalization"] == {"valid_start": "null"}
    payload["claims"][0]["valid_start"] = None
    claims = extractor.extract(
        owner_id="user",
        session_id="s1",
        messages=[
            Message(
                sender_id="user",
                role="user",
                timestamp=ensure_ms("2027-05-01"),
                content="text",
            )
        ],
    )
    assert "model_null_time_normalization" not in claims[0].metadata


def test_json_client_retains_invalid_response_and_clears_prior_usage():
    backend = object.__new__(_QwenJsonClient)
    backend.model = "fake"
    backend.temperature = 0
    backend.json_mode = True
    backend.last_usage = {"total_tokens": 999}
    backend.last_response_text = "previous"
    response = SimpleNamespace(
        usage=SimpleNamespace(prompt_tokens=3, completion_tokens=2, total_tokens=5),
        choices=[SimpleNamespace(message=SimpleNamespace(content="invalid json"))],
    )
    backend.client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **_: response),
        )
    )
    with pytest.raises(ValueError):
        backend.call("system", "user")
    assert backend.last_response_text == "invalid json"
    assert backend.last_usage["total_tokens"] == 5

    def network_failure(**kwargs):
        raise RuntimeError("connection failed")

    backend.client.chat.completions.create = network_failure
    with pytest.raises(RuntimeError):
        backend.call("system", "user")
    assert backend.last_response_text is None
    assert backend.last_usage == {}


def test_query_compiler_handles_quoted_null_and_nullable_place_lists():
    compiler = object.__new__(QwenQueryCompiler)
    compiler.backend = SimpleNamespace(
        last_usage={},
        call=lambda *_: {
            "temporal_intent": "as_of",
            "query_time": "2025-05-15",
            "knowledge_time": "null",
            "interval_start": "null",
            "interval_end": "null",
            "place": {"name": "Hangzhou", "aliases": None, "hierarchy": None},
        },
    )
    frame = compiler.compile(owner_id="u", query="What did I prefer in Hangzhou on 2025-05-15?")
    assert frame.query_time == ensure_ms("2025-05-15")
    assert frame.knowledge_time is frame.interval_start is frame.interval_end is None
    assert frame.place.name == "Hangzhou"
    assert frame.place.aliases == frame.place.hierarchy == []
