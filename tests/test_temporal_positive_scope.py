from __future__ import annotations

import pytest

from stacmem.models import ClaimDraft
from stacmem.temporal_grounding import TemporalGrounder
from stacmem.time_utils import ensure_ms


def ground(source):
    draft = ClaimDraft(
        owner_id="u", subject="u", predicate="residence.current", object_value="London",
        source_content=source, assertion_time=ensure_ms("2025-10-01"),
        valid_start=ensure_ms("2025-01-01"),
    )
    return draft, TemporalGrounder().ground_and_apply(draft, mode="certificate_v2")


@pytest.mark.parametrize("source", [
    "On 2025-01-01 I joined Aster and moved to London [UNPARSED_ADJUNCT].",
    "On 2025-01-01 I joined Aster [UNPARSED_ADJUNCT] and moved to London.",
    "On 2025-01-01 I left Northwind and joined Aster [UNPARSED_ADJUNCT] "
    "and moved to London.",
    "On 2025-01-01, I moved to London [UNPARSED_ADJUNCT].",
])
def test_inheritance_requires_positive_proof_even_if_veto_detector_misses_everything(
    monkeypatch, source,
):
    monkeypatch.setattr("stacmem.temporal_grounding._account_evidence", lambda *a: ())
    draft, cert = ground(source)
    assert draft.valid_start is None and draft.valid_end is None
    assert cert.inheritance_proof == "unsupported_body"
    assert cert.reasons == ("cross_proposition_scope_not_proven",)


@pytest.mark.parametrize("source", [
    "On 2025-01-01 I joined Aster and moved to London.",
    "On 2025-01-01 I left Northwind and joined Aster and moved to London.",
    "On 2025-01-01, I moved from Paris to London.",
])
def test_complete_supported_body_has_an_auditable_positive_proof(source):
    draft, cert = ground(source)
    assert draft.valid_start == ensure_ms("2025-01-01")
    assert cert.inheritance_proof == "complete_body_v1"
    assert cert.binding_scope in {"introductory", "coordinated_events"}
