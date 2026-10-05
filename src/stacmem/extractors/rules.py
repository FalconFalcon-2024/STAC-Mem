"""Conservative query parser used by offline tests and fallback paths."""

from __future__ import annotations

import re

from stacmem.models import QueryFrame, TemporalIntent
from stacmem.time_utils import ensure_ms, now_ms

_ISO_DATE = re.compile(r"\b(20\d{2}-\d{2}-\d{2})\b")


class RuleQueryCompiler:
    name = "rule-query-v1"

    def compile(self, *, owner_id: str, query: str) -> QueryFrame:
        normalized = query.casefold()
        temporal_intent = TemporalIntent.UNSPECIFIED
        query_time = None
        expected = "one"
        if any(token in normalized for token in ("history", "ever", "历史", "都有哪些")):
            temporal_intent = TemporalIntent.HISTORY
            expected = "many"
        elif any(
            token in normalized
            for token in ("currently", "current", "right now", "现在", "目前", "当前")
        ):
            temporal_intent = TemporalIntent.CURRENT
            query_time = now_ms()
        else:
            match = _ISO_DATE.search(query)
            if match:
                temporal_intent = TemporalIntent.AS_OF
                query_time = ensure_ms(f"{match.group(1)}T12:00:00Z")
        return QueryFrame(
            raw_query=query,
            owner_id=owner_id,
            temporal_intent=temporal_intent,
            query_time=query_time,
            expected_cardinality=expected,
            metadata={"compiler": self.name},
        )
