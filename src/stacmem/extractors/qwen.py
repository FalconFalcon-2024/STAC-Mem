"""Qwen OpenAI-compatible structured claim and query compilers."""

from __future__ import annotations

import json
from typing import Any

from stacmem.models import (
    ClaimDraft,
    Message,
    Place,
    QueryFrame,
    TemporalIntent,
    UpdateKind,
)
from stacmem.providers import OpenAIJsonClient, endpoint_options
from stacmem.time_utils import ensure_ms, now_ms

CLAIM_SYSTEM = """You compile conversation memory into explicit claims.
Return JSON only. Preserve provenance and distinguish event time from message time.
Do not invent missing dates, locations, subjects, or values.

Output schema:
{
  "claims": [
    {
      "subject": "canonical entity",
      "predicate": "short canonical relation such as residence.current",
      "object_value": "value",
      "valid_start": "ISO-8601 or null",
      "valid_end": "ISO-8601 exclusive or null",
      "place": {
        "place_id": null,
        "name": null,
        "aliases": [],
        "hierarchy": ["country", "region", "city", "venue"],
        "latitude": null,
        "longitude": null,
        "radius_km": null,
        "role": "scope|event|origin|destination"
      },
      "confidence": 0.0,
      "update_kind": "assertion|transition|correction|retraction",
      "functional": true,
      "source_message_ids": ["message id"],
      "source_content": "exact supporting source span",
      "proposition_grounding": {
        "evidence_span": "exact source span containing the claimed value",
        "proposition_subject": "self, canonical entity, or unknown",
        "subject_alignment": "aligned|misaligned|unknown",
        "factuality": "asserted|nonfactual",
        "transition_entailment": true,
        "previous_value": "explicit prior value or null"
      }
    }
  ]
}

Rules:
- valid time is when the claim is true in the world, not when it was said.
- a correction says an earlier claim was wrong; a transition says reality changed.
- use stable state predicates: residence.current, employment.organization,
  employment.position, occupation.current is not allowed, and language.spoken.
- "left X and joined Y", "moved to Y", and equivalent state changes are transitions,
  not assertions; emit the new current state with update_kind="transition".
- do not create a separate *.previous claim merely to restate the old value when a
  transition already carries the old and new state in its source evidence.
- source_content and proposition_grounding.evidence_span must be exact substrings of a
  source message, not generated summaries. The evidence span must contain object_value.
- factuality applies only to the proposition supporting object_value. A modal or negation
  in another clause must not change an asserted proposition's factuality.
- subject_alignment is aligned only when the evidence span describes the claim subject.
  Do not assign a colleague's, relative's, quoted speaker's, or another object's change
  to the user.
- for a first-person statement made by the user, set subject exactly to the input owner_id.
- transition_entailment is true only when the evidence states a real directional change
  from a prior state to object_value. A contradiction, wish, hypothesis, quotation, or
  report is not a transition.
- do not collapse place-scoped preferences into one global value.
- use null for unknown interval bounds; do not replace unknown with message time.
- omit acknowledgements and facts only about the assistant remembering something.
"""

QUERY_SYSTEM = """Compile a memory question into a query frame. Return JSON only.
Schema:
{
  "target_subject": "entity or null",
  "target_predicate": "canonical relation or null",
  "target_confidence": 0.0,
  "temporal_intent": "current|as_of|history|interval|unspecified",
  "query_time": "ISO-8601 or null",
  "knowledge_time": "ISO-8601 or null",
  "interval_start": "ISO-8601 or null",
  "interval_end": "ISO-8601 or null",
  "place": null or {
    "place_id": null,
    "name": null,
    "aliases": [],
    "hierarchy": [],
    "latitude": null,
    "longitude": null,
    "radius_km": null,
    "role": "scope"
  },
  "expected_cardinality": "one|many"
}
Current means the time at which the question is asked. Do not invent a location.
Use employment.organization for an employer and employment.position for a job title.
"""


def parse_model_place(raw: Any) -> Place | None:
    """Normalize nullable list members commonly produced by JSON-mode models."""
    if not isinstance(raw, dict):
        return None
    allowed = {key: value for key, value in raw.items() if key in Place.model_fields}
    for key in ("aliases", "hierarchy"):
        value = allowed.get(key)
        if isinstance(value, list):
            allowed[key] = [str(item) for item in value if item is not None and str(item).strip()]
        elif value is None:
            allowed[key] = []
    return Place.model_validate(allowed)


def parse_model_time(value: Any) -> int | None:
    """Accept a JSON-mode model's quoted null without guessing unknown dates."""
    if isinstance(value, str) and value.strip().lower() == "null":
        return None
    return ensure_ms(value)


class _QwenJsonClient(OpenAIJsonClient):
    """Compatibility preset for callers using the original Qwen adapter."""

    def __init__(self, *, base_url=None, api_key_env=None, **kwargs: Any) -> None:
        super().__init__(**endpoint_options("qwen", base_url, api_key_env), **kwargs)


class QwenClaimExtractor:
    name = "qwen-claim-v1"

    def __init__(self, *, model: str = "qwen-plus", backend=None, **kwargs: Any) -> None:
        self.backend = backend or _QwenJsonClient(model=model, **kwargs)

    def extract(
        self,
        *,
        owner_id: str,
        session_id: str,
        messages: list[Message],
    ) -> list[ClaimDraft]:
        if not messages:
            return []
        rendered = []
        for index, message in enumerate(messages):
            message_id = message.message_id or f"{session_id}:{index}"
            rendered.append(
                {
                    "message_id": message_id,
                    "sender_id": message.sender_id,
                    "role": message.role,
                    "timestamp_ms": message.timestamp,
                    "content": message.text(),
                }
            )
        payload = self.backend.call(
            CLAIM_SYSTEM,
            json.dumps(
                {"owner_id": owner_id, "session_id": session_id, "messages": rendered},
                ensure_ascii=False,
            ),
        )
        anchor = max(message.timestamp for message in messages)
        output: list[ClaimDraft] = []
        for raw in payload.get("claims", []):
            if not isinstance(raw, dict):
                continue
            place = parse_model_place(raw.get("place"))
            normalized_nulls = {
                field: raw[field]
                for field in ("valid_start", "valid_end")
                if isinstance(raw.get(field), str) and raw[field].strip().lower() == "null"
            }
            output.append(
                ClaimDraft(
                    owner_id=owner_id,
                    subject=str(raw.get("subject") or owner_id),
                    predicate=str(raw.get("predicate") or "unknown"),
                    object_value=str(raw.get("object_value") or "").strip(),
                    assertion_time=anchor,
                    observed_at=anchor,
                    valid_start=parse_model_time(raw.get("valid_start")),
                    valid_end=parse_model_time(raw.get("valid_end")),
                    place=place,
                    confidence=float(raw.get("confidence", 0.7)),
                    update_kind=UpdateKind(raw.get("update_kind", "assertion")),
                    functional=raw.get("functional"),
                    source_session_id=session_id,
                    source_message_ids=[str(item) for item in raw.get("source_message_ids", [])],
                    source_content=str(raw.get("source_content") or ""),
                    extractor=f"{self.name}:{self.backend.model}",
                    metadata={
                        "usage": self.backend.last_usage,
                        **(
                            {"model_null_time_normalization": normalized_nulls}
                            if normalized_nulls
                            else {}
                        ),
                        "proposition_grounding": (
                            raw.get("proposition_grounding")
                            if isinstance(raw.get("proposition_grounding"), dict)
                            else {}
                        ),
                    },
                )
            )
        return [claim for claim in output if claim.object_value and claim.predicate != "unknown"]


class QwenQueryCompiler:
    name = "qwen-query-v1"

    def __init__(self, *, model: str = "qwen-plus", backend=None, **kwargs: Any) -> None:
        self.backend = backend or _QwenJsonClient(model=model, **kwargs)

    def compile(self, *, owner_id: str, query: str) -> QueryFrame:
        asked_at = now_ms()
        payload = self.backend.call(
            QUERY_SYSTEM,
            json.dumps(
                {"owner_id": owner_id, "query": query, "asked_at_ms": asked_at},
                ensure_ascii=False,
            ),
        )
        place_raw = payload.get("place")
        intent = TemporalIntent(payload.get("temporal_intent", "unspecified"))
        query_time = parse_model_time(payload.get("query_time"))
        if intent == TemporalIntent.CURRENT and query_time is None:
            query_time = asked_at
        return QueryFrame(
            raw_query=query,
            owner_id=owner_id,
            target_subject=_optional_str(payload.get("target_subject")),
            target_predicate=_optional_str(payload.get("target_predicate")),
            temporal_intent=intent,
            query_time=query_time,
            knowledge_time=parse_model_time(payload.get("knowledge_time")),
            interval_start=parse_model_time(payload.get("interval_start")),
            interval_end=parse_model_time(payload.get("interval_end")),
            place=parse_model_place(place_raw),
            expected_cardinality=payload.get("expected_cardinality", "one"),
            metadata={
                "compiler": self.name,
                "target_confidence": float(payload.get("target_confidence", 0.7)),
                "usage": self.backend.last_usage,
            },
        )


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    rendered = str(value).strip()
    return rendered or None
