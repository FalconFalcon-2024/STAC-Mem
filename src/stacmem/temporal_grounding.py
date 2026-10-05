"""Source-grounded validation and normalization of claim valid-time intervals."""

from __future__ import annotations

import datetime as dt
import re
from dataclasses import asdict, dataclass
from typing import Any, Literal

from .models import ClaimDraft, UpdateKind, normalize_text
from .source_spans import SourceSpan, proposition_link, proposition_spans
from .temporal_evidence import DETECTOR as EVIDENCE_DETECTOR
from .temporal_evidence import EvidenceAccount, TemporalEvidence, account_temporal_evidence

TemporalDirection = Literal[
    "lower_bound", "upper_bound", "closed_interval", "observation", "none", "ambiguous"
]
TemporalConsistency = Literal["consistent", "corrected", "insufficient", "unsupported", "ambiguous"]
TemporalGroundingMode = Literal["certificate_v1", "certificate_v2"]

_DATE = r"\d{4}-\d{2}-\d{2}"
_ON_DATE = re.compile(rf"\bon\s+(?P<date>{_DATE})\b", re.IGNORECASE)
_AS_OF_DATE = re.compile(rf"\bas\s+of\s+(?P<date>{_DATE})\b|截至\s*(?P<zh>{_DATE})", re.I)
_STILL = re.compile(r"\bstill\b|仍然|仍旧|仍在|依然", re.I)
_DAY_MS = 86_400_000
_INTERVAL_PATTERNS = (
    re.compile(
        rf"\bfrom\s+(?P<start>{_DATE})\s+(?:to|until)\s+(?P<end>{_DATE})\b",
        re.IGNORECASE,
    ),
    re.compile(rf"从\s*(?P<start>{_DATE})\s*(?:到|至)\s*(?P<end>{_DATE})"),
)
_INTERVAL_PATTERNS_V2 = (
    *_INTERVAL_PATTERNS,
    re.compile(
        rf"\bbetween\s+(?P<start>{_DATE})\s+and\s+(?P<end>{_DATE})\b",
        re.IGNORECASE,
    ),
    re.compile(
        rf"(?P<start>{_DATE})\s*(?:到|至)\s*(?P<end>{_DATE})\s*(?:期间|之间)?"
    ),
)
_LOWER_PATTERNS = (
    re.compile(
        rf"\b(?:since|from|starting|beginning)\s+(?:on\s+)?(?P<date>{_DATE})\b",
        re.IGNORECASE,
    ),
    re.compile(rf"(?:从|自)\s*(?P<date>{_DATE})\s*(?:起|开始|以来)"),
    re.compile(rf"(?P<date>{_DATE})\s*(?:起|之后|以后)"),
)
_UPPER_PATTERNS = (
    re.compile(
        rf"\b(?:before|until|prior\s+to)\s+(?P<date>{_DATE})\b",
        re.IGNORECASE,
    ),
    re.compile(rf"(?P<date>{_DATE})\s*(?:之前|以前)"),
    re.compile(rf"(?:截至|直到)\s*(?P<date>{_DATE})"),
)

_CLAUSE_BOUNDARY = re.compile(r"[;,\uff0c\uff1b.!?\u3002\uff01\uff1f]")
_SENTENCE_BOUNDARY = re.compile(r"[.!?\u3002\uff01\uff1f\n]")
_CONTRAST_BOUNDARY = re.compile(
    r"\b(?:but|yet|however|instead)\b|\u4f46\u662f|\u7136\u800c|\u800c\u662f",
    re.IGNORECASE,
)
_DEPARTURE_PATTERN_V1 = re.compile(
    r"(?:\b(?:left|leave|leaving|departed|quit|resigned|stopped\s+working|"
    r"ended\s+(?:my\s+)?(?:job|work|employment)|"
    r"finished\s+(?:my\s+)?(?:job|work|employment|final\s+week))\b|"
    r"\u7ed3\u675f|\u79bb\u5f00|\u79bb\u804c|\u8f9e\u53bb|\u4e0d\u518d)",
    re.IGNORECASE,
)
_EVENT_ACTION = re.compile(
    r"\b(?:join(?:ed|ing)?|mov(?:ed|ing)|relocat(?:ed|ing)|"
    r"(?:started|starting|began)\s+working|switched|changed|became|promoted)\b|"
    r"加入|搬到|迁居到|转到|成为|正式到|入职",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class TemporalGroundingCertificate:
    detector: str
    direction: TemporalDirection
    support_span: str
    temporal_cue_span: str
    value_support_span: str
    parsed_start: int | None
    parsed_end: int | None
    original_valid_start: int | None
    original_valid_end: int | None
    normalized_valid_start: int | None
    normalized_valid_end: int | None
    object_grounded: bool
    consistency: TemporalConsistency
    normalization_applied: bool
    reasons: tuple[str, ...]
    temporal_cue_codepoint_span: tuple[int, int] | None = None
    value_support_codepoint_span: tuple[int, int] | None = None
    binding_scope: str = "unresolved"
    binding_scope_codepoint_span: tuple[int, int] | None = None
    unresolved_temporal_signal_spans: tuple[str, ...] = ()
    unresolved_temporal_signal_codepoint_spans: tuple[tuple[int, int], ...] = ()
    temporal_evidence_detector: str | None = None
    temporal_evidence_accounting: tuple[dict[str, Any], ...] = ()
    inheritance_proof: str = "not_required"

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class _TemporalCue:
    direction: Literal["lower_bound", "upper_bound", "closed_interval", "observation"]
    span: str
    start: int | None
    end: int | None
    offset: int

    @property
    def finish(self) -> int:
        return self.offset + len(self.span)


@dataclass(frozen=True)
class _ValueSupport:
    span: str
    offset: int
    detector: Literal["extractor_evidence", "source_clause"]
    bounds: tuple[int, int]


@dataclass(frozen=True)
class _CueAssociation:
    cues: tuple[_TemporalCue, ...] = ()
    kind: Literal["local", "introductory", "coordinated_events", "unresolved"] = "unresolved"
    bounds: tuple[int, int] | None = None
    blocked_signals: tuple[TemporalEvidence, ...] = ()
    evidence_accounts: tuple[EvidenceAccount, ...] = ()
    inheritance_proof: str = "not_required"


class TemporalGrounder:
    """Normalize explicit date-direction cues without using transaction recency."""

    detector = "temporal-grounding-v1"

    def ground(
        self,
        draft: ClaimDraft,
        *,
        mode: TemporalGroundingMode = "certificate_v1",
    ) -> TemporalGroundingCertificate:
        if mode not in {"certificate_v1", "certificate_v2"}:
            raise ValueError(f"unsupported temporal grounding mode: {mode}")
        source = draft.source_content
        value_support = _locate_value_support(draft, source, unique=mode == "certificate_v2")
        object_grounded = value_support is not None
        cues = _extract_cues(source, mode=mode)
        has_source_cues = bool(cues)
        association = _CueAssociation()
        if mode == "certificate_v2" and value_support is not None:
            association = _associate_cues(source, value_support, cues, draft.object_value)
            cues = list(association.cues)
        original = (draft.valid_start, draft.valid_end)
        if not cues:
            blocked = bool(association.blocked_signals)
            return self._certificate(
                draft,
                direction="none",
                consistency="unsupported" if has_source_cues or blocked else "insufficient",
                object_grounded=object_grounded,
                support_span=value_support.span if value_support else "",
                value_support_span=value_support.span if value_support else "",
                mode=mode,
                reasons=("cross_proposition_scope_not_proven"
                         if association.inheritance_proof == "unsupported_body"
                         else "independent_temporal_signal_blocks_binding" if blocked
                         else "temporal_cue_not_colocated_with_value_evidence" if has_source_cues
                         else "no_supported_iso_date_direction",),
                value_bounds=value_support.bounds if value_support else None,
                association=association,
            )
        if len(cues) != 1:
            return self._certificate(
                draft,
                direction="ambiguous",
                consistency="ambiguous",
                object_grounded=object_grounded,
                mode=mode,
                reasons=("multiple_temporal_directions",),
                association=association,
            )

        cue = cues[0]
        if not object_grounded:
            return self._certificate(
                draft,
                direction=cue.direction,
                consistency="unsupported",
                object_grounded=False,
                support_span=cue.span,
                temporal_cue_span=cue.span,
                parsed_start=cue.start,
                parsed_end=cue.end,
                mode=mode,
                reasons=("claim_value_not_grounded_in_temporal_source",),
            )

        assert value_support is not None
        if not _same_sentence(source, cue.offset, value_support.offset) or (
            _CONTRAST_BOUNDARY.search(
                source[min(cue.offset, value_support.offset):max(cue.offset, value_support.offset)]
            )
        ):
            return self._certificate(
                draft,
                direction=cue.direction,
                consistency="unsupported",
                object_grounded=True,
                support_span=value_support.span,
                temporal_cue_span=cue.span,
                value_support_span=value_support.span,
                parsed_start=cue.start,
                parsed_end=cue.end,
                mode=mode,
                reasons=(
                    "temporal_cue_not_colocated_with_value_evidence",
                    f"value_support_{value_support.detector}",
                ),
            )

        bound_cue, binding_reasons = _bind_cue_to_proposition(
            cue,
            value_support,
            object_value=draft.object_value,
            mode=mode,
        )

        if (
            mode == "certificate_v2"
            and bound_cue.direction == "upper_bound"
            and transition_value_role(value_support.span, draft.object_value) == "new"
            and not _direct_endpoint_support(cue, value_support, draft.object_value)
        ):
            # This guard is independent of segmentation: even a missed connector
            # cannot turn an arrival event into a state that ends before arrival.
            return self._certificate(
                draft, direction="upper_bound", consistency="unsupported", object_grounded=True,
                support_span=value_support.span, temporal_cue_span=cue.span,
                value_support_span=value_support.span, parsed_end=bound_cue.end, mode=mode,
                reasons=("new_value_upper_bound_requires_independent_endpoint_evidence",),
                cue_bounds=(cue.offset, cue.finish), value_bounds=value_support.bounds,
                association=association,
            )

        if association.blocked_signals:
            return self._certificate(
                draft, direction=bound_cue.direction, consistency="unsupported",
                object_grounded=True, mode=mode,
                support_span=value_support.span, temporal_cue_span=cue.span,
                value_support_span=value_support.span,
                parsed_start=bound_cue.start, parsed_end=bound_cue.end,
                reasons=("independent_temporal_signal_blocks_binding",),
                cue_bounds=(cue.offset, cue.finish), value_bounds=value_support.bounds,
                association=association,
            )

        if bound_cue.direction == "observation" and draft.update_kind in {
            UpdateKind.CORRECTION, UpdateKind.RETRACTION
        }:
            return self._certificate(
                draft, direction="observation", consistency="unsupported",
                object_grounded=True, mode=mode,
                reasons=("dated_destructive_update_not_a_positive_observation",),
            )

        normalized = (bound_cue.start, bound_cue.end)
        if mode == "certificate_v2" and not _is_positive_interval(*normalized):
            return self._certificate(
                draft,
                direction=bound_cue.direction,
                consistency="unsupported",
                object_grounded=True,
                support_span=value_support.span,
                temporal_cue_span=cue.span,
                value_support_span=value_support.span,
                parsed_start=bound_cue.start,
                parsed_end=bound_cue.end,
                mode=mode,
                reasons=(
                    f"source_entails_{bound_cue.direction}",
                    f"value_support_{value_support.detector}",
                    *binding_reasons,
                    "normalization_rejected_non_positive_interval",
                ),
                association=association,
            )
        changed = normalized != original
        return self._certificate(
            draft,
            direction=bound_cue.direction,
            consistency="corrected" if changed else "consistent",
            object_grounded=True,
            support_span=value_support.span,
            temporal_cue_span=cue.span,
            value_support_span=value_support.span,
            parsed_start=bound_cue.start,
            parsed_end=bound_cue.end,
            normalized_start=bound_cue.start,
            normalized_end=bound_cue.end,
            mode=mode,
            reasons=(
                f"source_entails_{bound_cue.direction}",
                f"value_support_{value_support.detector}",
                *binding_reasons,
                f"temporal_scope:{association.kind}" if mode == "certificate_v2"
                else "legacy_sentence_scope",
                "normalized_model_interval" if changed else "model_interval_consistent",
            ),
            cue_bounds=(cue.offset, cue.finish),
            value_bounds=value_support.bounds,
            association=association,
        )

    def ground_and_apply(
        self,
        draft: ClaimDraft,
        *,
        mode: TemporalGroundingMode = "certificate_v1",
    ) -> TemporalGroundingCertificate:
        certificate = self.ground(draft, mode=mode)
        draft.metadata = {
            **draft.metadata,
            "temporal_certificate": certificate.model_dump(),
        }
        if certificate.normalization_applied:
            draft.metadata = {
                **draft.metadata,
                "original_temporal_bounds": {
                    "valid_start": draft.valid_start,
                    "valid_end": draft.valid_end,
                },
                "temporal_normalizer": certificate.detector,
            }
            draft.valid_start = certificate.normalized_valid_start
            draft.valid_end = certificate.normalized_valid_end
        if certificate.direction == "observation" and certificate.consistency in {
            "consistent", "corrected"
        }:
            draft.metadata = {
                **draft.metadata,
                "temporal_observation": {
                    "kind": "holds_at", "grain": "day",
                    "support_span": certificate.value_support_span,
                    "temporal_cue_span": certificate.temporal_cue_span,
                    "original": {
                        "valid_start": certificate.original_valid_start,
                        "valid_end": certificate.original_valid_end,
                        "update_kind": draft.update_kind.value,
                    },
                    "normalizer": "observation-contract-v2",
                    "asserts_onset": False, "asserts_termination": False,
                },
            }
            draft.update_kind = UpdateKind.ASSERTION
        if mode == "certificate_v2":
            draft.metadata = {
                **draft.metadata,
                "temporal_eligibility": {
                    "version": "evidence-floor-v1",
                    "kind": "unknown_onset" if (
                        draft.valid_start is None and draft.valid_end is None
                    ) else "source_bounded",
                    "eligible_from": draft.assertion_time if (
                        draft.valid_start is None and draft.valid_end is None
                    ) else draft.valid_start,
                    "asserts_onset": draft.valid_start is not None
                    and certificate.direction != "observation",
                },
            }
        return certificate

    def _certificate(
        self,
        draft: ClaimDraft,
        *,
        direction: TemporalDirection,
        consistency: TemporalConsistency,
        object_grounded: bool,
        support_span: str = "",
        temporal_cue_span: str = "",
        value_support_span: str = "",
        parsed_start: int | None = None,
        parsed_end: int | None = None,
        normalized_start: int | None = None,
        normalized_end: int | None = None,
        mode: TemporalGroundingMode = "certificate_v1",
        reasons: tuple[str, ...] = (),
        cue_bounds: tuple[int, int] | None = None,
        value_bounds: tuple[int, int] | None = None,
        association: _CueAssociation | None = None,
    ) -> TemporalGroundingCertificate:
        has_normalized_bounds = consistency in {"consistent", "corrected"}
        final_start = normalized_start if has_normalized_bounds else None
        final_end = normalized_end if has_normalized_bounds else None
        applied = (draft.valid_start, draft.valid_end) != (final_start, final_end)
        return TemporalGroundingCertificate(
            detector=(self.detector if mode == "certificate_v1" else "temporal-grounding-v2"),
            direction=direction,
            support_span=support_span,
            temporal_cue_span=temporal_cue_span,
            value_support_span=value_support_span,
            parsed_start=parsed_start,
            parsed_end=parsed_end,
            original_valid_start=draft.valid_start,
            original_valid_end=draft.valid_end,
            normalized_valid_start=final_start,
            normalized_valid_end=final_end,
            object_grounded=object_grounded,
            consistency=consistency,
            normalization_applied=applied,
            reasons=reasons,
            temporal_cue_codepoint_span=cue_bounds,
            value_support_codepoint_span=value_bounds,
            binding_scope=association.kind if association else "unresolved",
            binding_scope_codepoint_span=association.bounds if association else None,
            inheritance_proof=association.inheritance_proof if association else "not_required",
            unresolved_temporal_signal_spans=tuple(
                signal.span for signal in association.blocked_signals
            ) if association else (),
            unresolved_temporal_signal_codepoint_spans=tuple(
                signal.bounds for signal in association.blocked_signals
            ) if association else (),
            temporal_evidence_detector=EVIDENCE_DETECTOR if mode == "certificate_v2" else None,
            temporal_evidence_accounting=tuple(
                account.model_dump() for account in association.evidence_accounts
            ) if association else (),
        )


def _extract_cues(
    source: str, *, mode: TemporalGroundingMode = "certificate_v1"
) -> list[_TemporalCue]:
    interval_patterns = (
        _INTERVAL_PATTERNS_V2 if mode == "certificate_v2" else _INTERVAL_PATTERNS
    )
    interval_matches = [
        match for pattern in interval_patterns for match in pattern.finditer(source)
    ]
    cues = []
    for match in interval_matches:
        start = _source_date_ms(match.group("start"), mode)
        end = _source_date_ms(match.group("end"), mode)
        if start is not None and end is not None:
            cues.append(_TemporalCue(
                direction="closed_interval", span=match.group(),
                start=start, end=end, offset=match.start(),
            ))
    cues = _deduplicate_cues(cues)
    if cues and mode == "certificate_v1":
        return cues

    for direction, patterns in (
        ("lower_bound", _LOWER_PATTERNS),
        ("upper_bound", _UPPER_PATTERNS),
    ):
        for pattern in patterns:
            for match in pattern.finditer(source):
                if any(cue.offset <= match.start() and match.end() <= cue.finish for cue in cues):
                    continue
                boundary = _source_date_ms(match.group("date"), mode)
                if boundary is None:
                    continue
                cues.append(
                    _TemporalCue(
                        direction=direction,
                        span=match.group(0),
                        start=boundary if direction == "lower_bound" else None,
                        end=boundary if direction == "upper_bound" else None,
                        offset=match.start(),
                    )
                )
    if mode == "certificate_v2":
        for pattern in (_ON_DATE, _AS_OF_DATE):
            for match in pattern.finditer(source):
                if match.groupdict().get("zh"):
                    # Chinese 'until/as of' is an upper bound unless the claim's
                    # own proposition states that the value still holds.
                    continue
                if any(cue.offset <= match.start() and match.end() <= cue.finish for cue in cues):
                    # 'starting on DATE' is an onset, not a separate observation.
                    if pattern is not _AS_OF_DATE:
                        continue
                    cues = [cue for cue in cues if cue.offset != match.start()]
                boundary = _source_date_ms(
                    match.groupdict().get("date") or match.group("zh"), mode,
                )
                if boundary is None:
                    continue
                cues.append(_TemporalCue(
                    direction="observation", span=match.group(), start=boundary,
                    end=boundary + _DAY_MS, offset=match.start(),
                ))
    return _deduplicate_cues(cues)


def _deduplicate_cues(cues: list[_TemporalCue]) -> list[_TemporalCue]:
    unique: dict[tuple[str, int | None, int | None, int], _TemporalCue] = {}
    for cue in cues:
        # Multiple patterns may describe the same literal cue; distinct occurrences
        # must remain distinct so two different events cannot share a date by accident.
        overlapping = next((old for old in unique.values() if (
            old.direction == cue.direction and old.start == cue.start and old.end == cue.end
            and old.offset <= cue.offset < old.finish
        )), None)
        if overlapping is None:
            unique.setdefault((cue.direction, cue.start, cue.end, cue.offset), cue)
    return sorted(unique.values(), key=lambda cue: cue.offset)


def _locate_value_support(
    draft: ClaimDraft, source: str, *, unique: bool = False
) -> _ValueSupport | None:
    object_value = draft.object_value.strip()
    if not object_value:
        return None

    value_pattern = re.escape(object_value)
    if object_value.isascii() and object_value[0].isalnum() and object_value[-1].isalnum():
        value_pattern = rf"(?<!\w){value_pattern}(?!\w)"
    matches = list(re.finditer(value_pattern, source, flags=re.IGNORECASE))
    if not matches or (unique and len(matches) != 1):
        return None
    match = matches[0]
    if unique:
        unit = next((span for span in proposition_spans(source)
                     if span.start <= match.start() < span.end), None)
        if unit is None:
            return None
        return _ValueSupport(unit.text, match.start(), "source_clause", (unit.start, unit.end))
    clause_start = _last_boundary(_CLAUSE_BOUNDARY, source, match.start())
    clause_end = _next_boundary(_CLAUSE_BOUNDARY, source, match.end())
    return _ValueSupport(
        span=source[clause_start:clause_end].strip(),
        offset=match.start(),
        detector="source_clause",
        bounds=(clause_start, clause_end),
    )


def _associate_cues(
    source: str, support: _ValueSupport, cues: list[_TemporalCue], object_value: str
) -> _CueAssociation:
    local = [cue for cue in cues if support.bounds[0] <= cue.offset < support.bounds[1]]
    current = SourceSpan(support.span, *support.bounds)
    accounts = list(_account_evidence(source, current, local, object_value))
    blockers = _unresolved_evidence(accounts)
    if local:
        return _CueAssociation(
            tuple(local), "unresolved" if blockers else "local",
            None if blockers else support.bounds, blockers, tuple(accounts),
        )
    if blockers:
        return _CueAssociation(blocked_signals=blockers, evidence_accounts=tuple(accounts))
    if not cues:
        return _CueAssociation(evidence_accounts=tuple(accounts))
    # Absence of a lexical veto is not evidence of shared temporal scope. Only
    # complete bodies in the deliberately small grammar can inherit a date.
    if not _supported_scope_body(support.span, object_value=object_value):
        return _CueAssociation(
            evidence_accounts=tuple(accounts), inheritance_proof="unsupported_body",
        )
    target_is_event = (
        transition_value_role(support.span, object_value) == "new"
        and _has_event_action(support.span)
    )
    preceding = [span for span in proposition_spans(source) if span.end <= support.bounds[0]]
    links = []
    for span in reversed(preceding):
        link = proposition_link(source, span, current)
        if link in {"barrier", "subordinate"}:
            break
        contained = [cue for cue in cues if span.start <= cue.offset < span.end]
        current_accounts = _account_evidence(source, span, contained, object_value)
        accounts.extend(current_accounts)
        blockers = _unresolved_evidence(current_accounts)
        if blockers:
            return _CueAssociation(blocked_signals=blockers, evidence_accounts=tuple(accounts))
        if contained:
            remainder = span.text
            for cue in contained:
                remainder = remainder.replace(cue.span, "")
            date_only = remainder.strip(" \t:，,") in {"", "期间", "之间", "在"}
            if date_only and not links:
                return _CueAssociation(
                    tuple(contained), "introductory", (span.start, support.bounds[1]),
                    evidence_accounts=tuple(accounts), inheritance_proof="complete_body_v1",
                )
            # An introductory On/Since date may dominate a connected event group.
            # A postfix date, an old state's endpoint, or another dated proposition
            # terminates this search rather than falling through to an earlier date.
            if (
                target_is_event and len(contained) == 1
                and _shareable_adjunct(contained[0], span, links if date_only else [*links, link])
                and (date_only or _supported_scope_body(remainder, event_only=True))
                and (date_only or _event_link(link, span, current))
            ):
                return _CueAssociation(
                    tuple(contained), "coordinated_events", (span.start, support.bounds[1]),
                    evidence_accounts=tuple(accounts), inheritance_proof="complete_body_v1",
                )
            return _CueAssociation(
                evidence_accounts=tuple(accounts), inheritance_proof="unsupported_body",
            )
        if not (
            target_is_event and _supported_scope_body(span.text, event_only=True)
            and _event_link(link, span, current)
        ):
            return _CueAssociation(
                evidence_accounts=tuple(accounts), inheritance_proof="unsupported_body",
            )
        links.append(link)
        current = span
    return _CueAssociation(evidence_accounts=tuple(accounts))


def _supported_scope_body(
    text: str, *, object_value: str | None = None, event_only: bool = False,
) -> bool:
    """Positive, full-clause proof; no free-form modifier or entity tail.

    Targets must match their literal value. Intervening event entities have a
    restricted proper-name shape, not arbitrary prose. Unknown structures retain
    unknown bounds; this is a contract, not a general natural-language parser.
    """
    name = r"(?-i:[A-Z][A-Za-z0-9'-]*)(?:\s+(?:Labs|Works|Health|Inc|Ltd|Corp))?"
    value = re.escape(object_value) if object_value is not None else name
    subject = r"(?:(?:I|we)\s+(?:am\s+scheduled\s+to\s+)?)?"
    event = (
        rf"(?:join(?:ed)?\s+{value}|(?:left|leave)\s+{value}|"
        rf"(?:moved|relocated)\s+(?:from\s+{name}\s+)?to\s+{value}|"
        rf"(?:started|began)\s+working\s+at\s+{value})"
    )
    state = (
        rf"(?:still\s+)?(?:work(?:ed)?\s+at|live(?:d)?\s+in)\s+{value}|"
        rf"my\s+(?:residence|address)\s+(?:is|was)\s+{value}|"
        rf"in\s+{name}\s+I\s+(?:prefer|changed\s+to)\s+{value}"
    )
    body = event if event_only else rf"(?:{event}|{state})"
    # A closed manner adjunct is distinct from an unparsed temporal adjunct.
    manner = r"(?:\s+by\s+train)?" if object_value is not None else ""
    if re.fullmatch(
        rf"{subject}{body}{manner}", text.strip(" \t\r\n.,:;!?"), re.I,
    ):
        return True
    if object_value is None:
        return False
    # Retain the already-supported explicit Chinese prefix/body contracts; no
    # arbitrary material is permitted between their grammatical constituents.
    zh_state = (
        rf"(?:\u6211|\u6211\u4eec)(?:\u4ecd\u7136|\u4ecd\u65e7)?"
        rf"(?:\u4f4f\u5728|\u5c45\u4f4f\u5728)\s*{value}|"
        rf"\u6211\u7684(?:\u4f4f\u5740|\u5c45\u4f4f\u5730)\u662f\s*{value}"
    )
    zh_transition = (
        rf"\u6211\u4ece\s*(?:{value}\s*\u79bb\u804c\u5e76\u52a0\u5165\s*{name}|"
        rf"{name}\s*\u79bb\u804c\u5e76\u52a0\u5165\s*{value})"
    )
    return bool(re.fullmatch(
        rf"(?:{zh_transition}|{zh_state})", text.strip(" \t\r\n\u3002\uff0c"), re.I,
    ))


def _account_evidence(
    source: str, span: SourceSpan, accepted: list[_TemporalCue], object_value: str,
) -> tuple[EvidenceAccount, ...]:
    return account_temporal_evidence(
        source, span, consumed_cues=[(cue.offset, cue.finish) for cue in accepted],
        object_value=object_value,
    )


def _unresolved_evidence(accounts: list[EvidenceAccount] | tuple[EvidenceAccount, ...]):
    return tuple(account.evidence for account in accounts if account.disposition == "unresolved")


def _has_event_action(span: str) -> bool:
    return bool(_EVENT_ACTION.search(span) or _DEPARTURE_PATTERN_V1.search(span))


def _event_link(link: str, left: SourceSpan, right: SourceSpan) -> bool:
    if link in {"coordination", "sequence"}:
        return True
    # Preserve subject-elliptical comma continuations, including the existing
    # Chinese departure/arrival construction, but not unrelated full sentences.
    return link == "clause" and (
        bool(_DEPARTURE_PATTERN_V1.search(left.text))
        or not re.match(r"(?:i|we|my|our)\b|我|我们", right.text, re.I)
    )


def _shareable_adjunct(cue: _TemporalCue, span: SourceSpan, links: list[str]) -> bool:
    if span.text[:cue.offset - span.start].strip():
        return False
    if cue.direction == "lower_bound":
        return "sequence" not in links
    return cue.direction == "observation" and bool(_ON_DATE.fullmatch(cue.span))


def _direct_endpoint_support(cue: _TemporalCue, support: _ValueSupport, value: str) -> bool:
    # A pure event-time 'before DATE' is not evidence that the resulting state
    # ended then. Require a literal occupancy/employment predicate tied to its value.
    state = (
        rf"(?:i\s+|we\s+)?(?:work(?:ed)?\s+at|live(?:d)?\s+in|stay(?:ed)?\s+in)"
        rf"\s+{re.escape(value)}"
    )
    endpoint = re.escape(cue.span)
    return bool(re.search(
        rf"(?:{state}\s+{endpoint}|{endpoint}\s+{state})(?!\w)", support.span, re.I
    ))


def _contains_value(span: str, object_value: str) -> bool:
    return normalize_text(object_value) in normalize_text(span)


def _same_sentence(source: str, first_offset: int, second_offset: int) -> bool:
    low, high = sorted((first_offset, second_offset))
    return _SENTENCE_BOUNDARY.search(source, low, high) is None


def _bind_cue_to_proposition(
    cue: _TemporalCue,
    support: _ValueSupport,
    *,
    object_value: str,
    mode: TemporalGroundingMode,
) -> tuple[_TemporalCue, tuple[str, ...]]:
    if mode == "certificate_v2" and cue.span.startswith("截至") and _STILL.search(support.span):
        return _TemporalCue(
            direction="observation", span=cue.span, start=cue.end,
            end=cue.end + _DAY_MS, offset=cue.offset,
        ), ("dated_state_is_day_observation",)
    if mode == "certificate_v2" and cue.direction == "observation":
        role = transition_value_role(support.span, object_value)
        # 'As of DATE ... still ...' reports a state, never an onset or departure.
        if _AS_OF_DATE.match(cue.span):
            return cue, ("dated_state_is_day_observation",)
        if role in {"prior", "new"}:
            direction = "upper_bound" if role == "prior" else "lower_bound"
            return _TemporalCue(
                direction=direction, span=cue.span, start=cue.start if role == "new" else None,
                end=cue.start if role == "prior" else None, offset=cue.offset,
            ), (f"dated_transition_{role}_value",)
        return cue, ("dated_state_is_day_observation",)
    if cue.direction != "lower_bound":
        return cue, ("temporal_direction_preserved",)
    if mode == "certificate_v1":
        if not _DEPARTURE_PATTERN_V1.search(support.span):
            return cue, ("temporal_direction_preserved",)
        role = "departure"
    else:
        role = transition_value_role(support.span, object_value)
        if role != "prior":
            reason = (
                "arrival_proposition_preserves_lower_bound"
                if role == "new"
                else "temporal_direction_preserved"
            )
            return cue, (reason,)
    return (
        _TemporalCue(
            direction="upper_bound",
            span=cue.span,
            start=None,
            end=cue.start,
            offset=cue.offset,
        ),
        ("departure_proposition_inverts_lower_bound",),
    )


def transition_value_role(
    span: str, object_value: str
) -> Literal["prior", "new", "unknown"]:
    """Ground the value's direction in the source clause, not the model's label."""
    value = re.escape(object_value.strip())
    if not value:
        return "unknown"
    boundary = r"(?!\w)"
    prior_english = re.search(
        rf"\b(?:from|left|leave|leaving|departed(?:\s+from)?|quit|"
        rf"resigned\s+from|moved\s+from|relocated\s+from|"
        rf"end(?:ed|ing)?\s+(?:my\s+)?tenure\s+at)\s+(?:the\s+)?{value}{boundary}",
        span, re.IGNORECASE,
    )
    prior_chinese = re.search(
        rf"(?:\u79bb\u5f00|\u79bb\u804c|\u4ece|\u7ed3\u675f\u4e86\u5728)"
        rf"\s*{value}{boundary}", span,
    )
    new_english = re.search(
        rf"\b(?:join(?:s|ed|ing)?|accepted\s+(?:a\s+)?(?:position|job)\s+at|"
        rf"took\s+(?:a\s+)?job\s+at|(?:started|began)\s+working\s+at|"
        rf"(?:moved|moving|relocated|relocating|switched|changed|promoted)\s+to|"
        rf"now\s+work(?:ing)?\s+at|became)\s+(?:the\s+)?{value}{boundary}",
        span, re.IGNORECASE,
    )
    new_chinese = re.search(
        rf"(?:\u52a0\u5165|\u6b63\u5f0f\u5230|\u642c\u5230|\u8fc1\u5c45\u5230|"
        rf"\u8f6c\u5230|\u6210\u4e3a)\s*{value}{boundary}", span,
    )
    directional_to = re.search(rf"\bto\s+{value}{boundary}", span, re.IGNORECASE)
    directional_cue = re.search(
        r"\b(?:from|moved|relocated|switched|changed|promoted)\b", span,
        re.IGNORECASE,
    )
    prior = bool(prior_english or prior_chinese)
    new = bool(new_english or new_chinese or (directional_to and directional_cue))
    if bool(prior) != bool(new):
        return "prior" if prior else "new"
    return "unknown"


def _is_positive_interval(start: int | None, end: int | None) -> bool:
    return start is None or end is None or start < end


def _last_boundary(pattern: re.Pattern[str], source: str, offset: int) -> int:
    last = 0
    for match in pattern.finditer(source, 0, offset):
        last = match.end()
    return last


def _next_boundary(pattern: re.Pattern[str], source: str, offset: int) -> int:
    match = pattern.search(source, offset)
    return match.start() if match is not None else len(source)


def _date_ms(value: str) -> int:
    parsed = dt.date.fromisoformat(value)
    timestamp = dt.datetime.combine(parsed, dt.time.min, tzinfo=dt.UTC)
    return int(timestamp.timestamp() * 1000)


def _source_date_ms(value: str, mode: TemporalGroundingMode) -> int | None:
    try:
        return _date_ms(value)
    except ValueError:
        if mode == "certificate_v1":
            raise
        # Invalid calendar dates remain lexical evidence for the binding veto.
        return None
