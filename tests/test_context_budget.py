from __future__ import annotations

from stacmem.context_budget import estimate_tokens
from stacmem.models import Claim, ClaimDraft, EvidencePack, QueryFrame, ScoredClaim
from stacmem.resolver import compose_context


def test_multilingual_estimate_does_not_treat_cjk_like_four_ascii_characters():
    assert estimate_tokens("测试上下文") == 5
    assert estimate_tokens("abcdefgh") == 2
    assert estimate_tokens("hello, world") >= 4


def test_context_budget_is_visible_and_truncates_by_estimated_tokens():
    claims = []
    for index in range(4):
        draft = ClaimDraft(
            owner_id="u",
            subject="u",
            predicate="employment.organization",
            object_value=f"Company {index}",
            assertion_time=index + 1,
            source_content="中文证据" * 20,
        )
        claims.append(ScoredClaim(claim=Claim.from_draft(draft), score={}))
    pack = EvidencePack(query=QueryFrame(raw_query="q", owner_id="u"), claims=claims)
    pack.context = compose_context(pack, 80)
    assert pack.diagnostics["context_budget_method"] == "multilingual-estimate-v1"
    assert pack.diagnostics["context_estimated_tokens"] <= 80
    assert pack.diagnostics["context_truncated"] is True
    assert pack.context.count("[C") < len(claims)
