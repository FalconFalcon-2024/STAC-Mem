import json
from types import SimpleNamespace

from stacmem.contracts import ContractQueryCompiler


def test_replayed_current_query_binds_to_supplied_clock():
    calls = []

    def call(system, user):
        calls.append(json.loads(user))
        return {
            "target_predicate": "employment.organization", "temporal_intent": "current",
            "target_subject": "self", "query_time": "2099-01-01",
        }

    compiler = ContractQueryCompiler(SimpleNamespace(call=call, last_usage={}))
    frame = compiler.compile(
        owner_id="u", query="Where do I currently work?", asked_at=1722470400000
    )
    assert frame.query_time == 1722470400000
    assert calls[0]["asked_at_ms"] == 1722470400000
