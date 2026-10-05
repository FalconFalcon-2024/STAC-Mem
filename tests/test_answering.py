from __future__ import annotations

from stacmem.answering import answer_evidence


class Backend:
    def __init__(self, response):
        self.response = response
        self.calls = 0
        self.last_usage = {"total_tokens": 1}

    def call(self, *args):
        self.calls += 1
        return self.response


def evidence(values=(), warnings=()):
    return {
        "query": {"raw_query": "Where?"},
        "context": "test evidence",
        "claims": [{"claim": {"object_value": v}} for v in values],
        "warnings": list(warnings),
    }


def test_empty_evidence_never_calls_answer_model():
    backend = Backend(None)
    result = answer_evidence(evidence(), backend)
    assert result["abstain"] and result["answer"] and backend.calls == 0


def test_warning_is_not_silently_resolved_by_model():
    backend = Backend(None)
    result = answer_evidence(evidence(["Bern"], ["unresolved conflict"]), backend)
    assert result["abstain"] and backend.calls == 0


def test_empty_abstention_text_has_fallback_but_raw_is_preserved():
    raw = {"answer": "", "values": [], "abstain": True}
    result = answer_evidence(evidence(["Bern"]), Backend(raw))
    assert result["answer"] and result["abstain"]
    assert result["model_answer"] == raw


def test_unselected_values_cannot_be_displayed_as_supported():
    raw = {"answer": "Paris", "values": ["Paris"], "abstain": False}
    result = answer_evidence(evidence(["Bern"]), Backend(raw))
    assert result["abstain"] and result["model_answer"] == raw


def test_render_values_not_additional_model_prose():
    raw = {"answer": "Bern and I invented more facts", "values": ["Bern"], "abstain": False}
    result = answer_evidence(evidence(["Bern"]), Backend(raw))
    assert result["answer"] == "Bern" and result["model_answer"] == raw
