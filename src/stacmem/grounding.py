"""Proposition-level source grounding for admission and transition decisions."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any, Literal

from .models import Claim, ClaimDraft, UpdateKind, normalize_text
from .source_spans import proposition_link, proposition_spans
from .temporal_grounding import transition_value_role

Factuality = Literal["asserted", "nonfactual", "unsupported"]
SubjectAlignment = Literal["aligned", "misaligned", "unknown"]

_SENTENCE_BOUNDARY = re.compile(r"[.;!?。！？；\n]")
_CONTRAST = re.compile(r"\b(?:but|yet|however|instead)\b|但是|然而|而是|不过", re.I)
_QUESTION = re.compile(
    r"^(?:do|does|did|am|is|are|was|were|can|could|may|might|will|would|"
    r"should|have|has|had)\b|\b(?:wonder|ask)\s+(?:if|whether)\b|"
    r"是否|是不是|难道|吗[？?\s]*$",
    re.I,
)
_CORRECTION = re.compile(
    r"\b(?:correction|correcting|actually|mistaken|incorrect|wrong)\b|更正|纠正|说错|实际|其实",
    re.I,
)
_FIRST_PERSON = re.compile(r"\b(?:i|i'm|i've|my|me|we|our|us)\b", re.IGNORECASE)
_THIRD_PARTY = re.compile(
    r"\b(?:my|our)\s+(?:colleague|coworker|friend|sister|brother|manager|"
    r"spouse|partner|roommate|cousin|aunt|uncle|parent|mother|father|"
    r"supervisor|team\s+lead)\b",
    re.IGNORECASE,
)
_CHINESE_FIRST_PERSON = ("我", "本人", "我们")
_CHINESE_THIRD_PARTY = (
    "我的同事",
    "我同事",
    "我的朋友",
    "我朋友",
    "我的姐姐",
    "我的妹妹",
    "我的哥哥",
    "我的弟弟",
    "我的经理",
    "我的主管",
    "我的伴侣",
    "我的室友",
    "我父亲",
    "我母亲",
)

_NONFACTUAL_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "hypothetical_scope",
        re.compile(
            r"\b(?:if|unless|would|suppose|supposing|imagine|imagining|counterfactual|"
            r"hypothetical|were\s+i\s+to|were\s+we\s+to)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "negated_proposition",
        re.compile(
            r"\b(?:not|never|didn't|did\s+not|don't|do\s+not|isn't|is\s+not|"
            r"wasn't|was\s+not|aren't|can't|cannot|won't|hasn't|haven't|no\s+longer)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "uncertain_or_intended",
        re.compile(
            r"\b(?:may|might|could|possibly|perhaps|intend(?:s|ed)?\s+to|"
            r"plan(?:s|ned|ning)?\s+to|hope(?:s|d)?\s+to|aspir(?:e|es|ed|ation)|"
            r"future\s+(?:day|possibility|option))\b",
            re.IGNORECASE,
        ),
    ),
    (
        "quoted_reported_or_fictional",
        re.compile(
            r"\b(?:quote|quotation|rumou?r|report(?:s|ed)?|hearsay|allegedly|"
            r"fictional|screenplay|roleplay|role-play|simulation|example)\b",
            re.IGNORECASE,
        ),
    ),
)
_CHINESE_NONFACTUAL: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("hypothetical_scope", ("如果", "假如", "假设", "若", "倘若", "设想")),
    ("negated_proposition", ("没有", "并未", "从未", "不是", "不再")),
    ("uncertain_or_intended", ("可能", "也许", "或许", "计划", "打算", "希望", "意向")),
    (
        "quoted_reported_or_fictional",
        ("传言", "报道", "听闻", "据传", "虚构", "剧本", "角色扮演", "模拟", "例子"),
    ),
)
_DENIAL_FOLLOWUP = re.compile(
    r"\b(?:that|the)\s+(?:report|claim|story|statement)\s+(?:is|was)\s+"
    r"(?:false|untrue|incorrect)\b",
    re.IGNORECASE,
)
_NONFACTUAL_FRAME = re.compile(
    r"^(?:in|within|for)\s+(?:a\s+|the\s+)?(?:screenplay|simulation|roleplay|"
            r"role-play|fictional\s+profile|hypothetical(?:\s+\w+)?|example)\b",
    re.IGNORECASE,
)
_TRANSITION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "employment_change",
        re.compile(
            r"\b(?:depart(?:ed|ing)?|resign(?:ed|ing)?|quit|ended?\s+(?:my\s+)?tenure|"
            r"took|accepted|began|onboarded|hired|join(?:s|ed|ing)?|left|"
            r"start(?:s|ed|ing)?|new\s+employer|new\s+role)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "state_relocation",
        re.compile(
            r"\b(?:move|moved|moving|relocat(?:e|ed|ing)|settled?|established|transferred?|"
            r"shifted?|switched?|advanced?|elevated?|promoted?|became|changed?|"
            r"replaced?|succeeded?)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "directional_construction",
        re.compile(r"\b(?:from\b.+\bto|no\s+longer\b.+\bnow|gave\s+way\s+to)\b", re.I),
    ),
)
_CHINESE_TRANSITION: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "employment_change",
        ("辞职", "辞去", "离任", "入职", "任职", "就职", "受聘", "转岗"),
    ),
    (
        "state_relocation",
        ("搬迁", "迁往", "定居", "转至", "转到", "晋升", "升任", "调到", "转移", "改放"),
    ),
    ("directional_construction", ("由", "变为", "换为", "取代", "不再", "如今")),
)

_PAST_ONLY_STATE = re.compile(
    r"\b(?:i|we)\s+(?:(?:previously|formerly|once)\s+)?"
    r"(?:used\s+to\s+(?:work|live)|worked|lived|had\s+worked|had\s+lived|"
    r"was\s+(?:working|living|employed)|were\s+(?:working|living|employed))\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class PropositionCertificate:
    detector: str
    support_span: str
    support_unit_index: int | None
    object_grounded: bool
    subject_alignment: SubjectAlignment
    factuality: Factuality
    factuality_reasons: tuple[str, ...]
    transition_entailment: bool
    transition_confidence: float
    transition_reasons: tuple[str, ...]
    transition_role: Literal["prior", "new", "unknown"] = "unknown"
    current_state_eligible: bool = True
    current_state_reasons: tuple[str, ...] = ()
    structured_hint_used: bool = False
    source_context_span: str = ""
    support_codepoint_span: tuple[int, int] | None = None

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


class PropositionGrounder:
    """Build a deterministic, auditable certificate over the claim-supporting span."""

    detector = "deterministic-proposition-grounding-v4"

    def ground(
        self,
        draft: ClaimDraft,
        existing: Sequence[Claim] = (),
    ) -> PropositionCertificate:
        source = draft.source_content
        units = split_proposition_units(source)
        support_span, support_index, object_grounded = select_support_span(
            units, draft.object_value
        )
        hints = _validated_hints(draft, source)
        hint_used = bool(hints)
        context, support_offsets = _support_context(source, support_index)
        alignment = infer_subject_alignment(draft, context, hints)
        factuality, factuality_reasons = infer_factuality(
            draft,
            support_span,
            units,
            support_index,
            hints,
            object_grounded,
            context=context,
            source=source,
            support_offsets=support_offsets,
        )
        role = transition_value_role(support_span, draft.object_value)
        state_reasons = []
        if draft.functional and role == "prior":
            state_reasons.append("object_is_prior_transition_value")
        if draft.functional and role == "unknown" and _has_transition_signal(support_span):
            state_reasons.append("transition_destination_unverified")
        temporal = draft.metadata.get("temporal_certificate", {})
        supported_end = (
            isinstance(temporal, dict)
            and temporal.get("consistency") in {"consistent", "corrected"}
            and draft.valid_end is not None
        )
        if draft.functional and _past_only_support(source, support_index) and not supported_end:
            state_reasons.append("past_only_state_without_supported_end")
        transition, confidence, transition_reasons = infer_transition(
            draft,
            existing,
            support_span,
            source,
            alignment,
            factuality,
            hints,
            object_grounded,
            role=role,
        )
        return PropositionCertificate(
            detector=self.detector,
            support_span=support_span,
            support_unit_index=support_index,
            object_grounded=object_grounded,
            subject_alignment=alignment,
            factuality=factuality,
            factuality_reasons=tuple(factuality_reasons),
            transition_entailment=transition,
            transition_confidence=round(confidence, 3),
            transition_reasons=tuple(transition_reasons),
            transition_role=role,
            current_state_eligible=not state_reasons,
            current_state_reasons=tuple(state_reasons),
            structured_hint_used=hint_used,
            source_context_span=context,
            support_codepoint_span=support_offsets,
        )


def split_proposition_units(source: str) -> list[str]:
    return [span.text for span in proposition_spans(source)]


def _past_only_support(source: str, index: int | None) -> bool:
    if index is None:
        return False
    spans = proposition_spans(source)
    current = spans[index]
    while True:
        if _PAST_ONLY_STATE.search(current.text):
            return True
        if index == 0 or not re.match(r"(?:at|in)\b", current.text, re.I):
            return False
        previous = spans[index - 1]
        if proposition_link(source, previous, current) not in {"coordination", "sequence"}:
            return False
        current, index = previous, index - 1


def select_support_span(units: list[str], object_value: str) -> tuple[str, int | None, bool]:
    matches = [
        (index, unit)
        for index, unit in enumerate(units)
        if _contains_value(unit, object_value)
    ]
    if not matches:
        return "", None, False
    index, unit = min(matches, key=lambda item: (len(item[1]), item[0]))
    return unit, index, True


def infer_subject_alignment(
    draft: ClaimDraft,
    support_span: str,
    hints: dict[str, Any],
) -> SubjectAlignment:
    hinted = str(hints.get("subject_alignment") or "").casefold()
    if hinted in {"misaligned", "unknown"}:
        return hinted  # type: ignore[return-value]
    proposition_subject = normalize_text(str(hints.get("proposition_subject") or ""))
    if proposition_subject and proposition_subject not in {
        "self", "user", normalize_text(draft.owner_id), normalize_text(draft.subject)
    }:
        return "misaligned"

    normalized = normalize_text(support_span)
    if not normalized:
        return "unknown"
    if _THIRD_PARTY.search(support_span) or any(
        marker in support_span for marker in _CHINESE_THIRD_PARTY
    ):
        return "misaligned"

    aliases = _subject_aliases(draft)
    if any(alias and alias in normalized for alias in aliases):
        return "aligned"
    if draft.subject == draft.owner_id and (_FIRST_PERSON.search(support_span) or any(
        marker in support_span for marker in _CHINESE_FIRST_PERSON
    )):
        return "aligned"

    if _different_object_subject(normalized, aliases):
        return "misaligned"
    return "unknown"


def infer_factuality(
    draft: ClaimDraft,
    support_span: str,
    units: list[str],
    support_index: int | None,
    hints: dict[str, Any],
    object_grounded: bool,
    *,
    context: str = "",
    source: str = "",
    support_offsets: tuple[int, int] | None = None,
) -> tuple[Factuality, list[str]]:
    hinted = str(hints.get("factuality") or "").casefold()
    if not object_grounded:
        return "unsupported", ["object_not_grounded_in_source"]

    reasons: list[str] = []
    context = context or support_span
    for reason, pattern in _NONFACTUAL_PATTERNS:
        if pattern.search(context):
            reasons.append(reason)
    for reason, markers in _CHINESE_NONFACTUAL:
        if any(marker in context for marker in markers):
            reasons.append(reason)
    if "?" in context or "？" in context or _QUESTION.search(context.strip(" \t\"“”")):
        reasons.append("interrogative_speech_act")
    if support_offsets and _quoted_proposition(source, draft, support_offsets):
        reasons.append("quoted_reported_or_fictional")
    if support_index is not None and support_index > 0:
        prefix = units[support_index - 1]
        if _NONFACTUAL_FRAME.search(prefix) or any(
            marker in prefix for marker in ("在虚构情节中", "在模拟中", "在假设练习中")
        ):
            reasons.append("nonfactual_frame")
    if support_index is not None and support_index + 1 < len(units):
        followup = units[support_index + 1]
        if _DENIAL_FOLLOWUP.search(followup) or any(
            marker in followup for marker in ("这不是真的", "该说法不实", "消息是假的")
        ):
            reasons.append("explicit_denial_followup")

    if draft.update_kind == UpdateKind.RETRACTION:
        if "negated_proposition" in reasons:
            reasons.remove("negated_proposition")
        else:
            reasons.append("retraction_not_grounded")
    if draft.update_kind == UpdateKind.CORRECTION and not _CORRECTION.search(context):
        reasons.append("correction_not_grounded")
    if draft.predicate.casefold().strip().startswith("plan."):
        reasons = [reason for reason in reasons if reason != "uncertain_or_intended"]
    if hinted in {"nonfactual", "unsupported"}:
        reasons.append(f"structured_hint:{hinted}")
    return ("nonfactual", list(dict.fromkeys(reasons))) if reasons else ("asserted", [])


def infer_transition(
    draft: ClaimDraft,
    existing: Sequence[Claim],
    support_span: str,
    source: str,
    alignment: SubjectAlignment,
    factuality: Factuality,
    hints: dict[str, Any],
    object_grounded: bool,
    *,
    role: Literal["prior", "new", "unknown"] = "unknown",
) -> tuple[bool, float, list[str]]:
    if draft.update_kind in {UpdateKind.CORRECTION, UpdateKind.RETRACTION}:
        return False, 0.0, [f"explicit_{draft.update_kind.value}"]
    if factuality != "asserted" or alignment != "aligned" or not object_grounded:
        return False, 0.0, ["blocked_by_proposition_certificate"]

    hinted = hints.get("transition_entailment")
    if hinted is False:
        return False, 0.0, ["structured_transition_veto"]
    if draft.functional and role != "new":
        return False, 0.0, [
            "object_is_prior_transition_value" if role == "prior"
            else "transition_destination_unverified"
        ]

    different = [
        old for old in existing if old.object_norm != normalize_text(draft.object_value)
    ]
    reasons = ["object_grounded"]
    score = 0.30
    if alignment == "aligned":
        score += 0.15
        reasons.append("subject_aligned")
    if any(_draft_is_newer(old, draft) for old in different):
        score += 0.20
        reasons.append("later_state_time")
    if any(_contains_value(support_span, old.object_value) for old in different):
        score += 0.15
        reasons.append("prior_value_grounded")

    evidence_window = support_span
    transition_signals: list[str] = []
    for reason, pattern in _TRANSITION_PATTERNS:
        if pattern.search(evidence_window):
            transition_signals.append(reason)
    for reason, markers in _CHINESE_TRANSITION:
        if any(marker in evidence_window for marker in markers):
            transition_signals.append(reason)
    if transition_signals:
        score += 0.35
        reasons.extend(f"semantic:{item}" for item in dict.fromkeys(transition_signals))

    entailed = bool(transition_signals) and (not different or score >= 0.75)
    if not entailed:
        reasons.append("transition_threshold_not_met")
    return entailed, min(score, 1.0), reasons


def _has_transition_signal(span: str) -> bool:
    span = re.sub(
        r"\bfrom\s+\d{4}-\d{2}-\d{2}\s+to\s+\d{4}-\d{2}-\d{2}\b",
        "", span, flags=re.IGNORECASE,
    )
    return any(pattern.search(span) for _, pattern in _TRANSITION_PATTERNS) or any(
        marker in span for _, markers in _CHINESE_TRANSITION for marker in markers
    )


def _validated_hints(draft: ClaimDraft, source: str) -> dict[str, Any]:
    raw = draft.metadata.get("proposition_grounding")
    if not isinstance(raw, dict):
        return {}
    evidence_span = str(raw.get("evidence_span") or "").strip()
    if evidence_span and normalize_text(evidence_span) not in normalize_text(source):
        return {}
    return dict(raw)


def _contains_value(text: str, value: str) -> bool:
    if not value.strip():
        return False
    escaped = re.escape(normalize_text(value))
    if value.isascii() and value[0].isalnum() and value[-1].isalnum():
        escaped = rf"(?<!\w){escaped}(?!\w)"
    return re.search(escaped, normalize_text(text)) is not None


def _support_context(source: str, index: int | None) -> tuple[str, tuple[int, int] | None]:
    if index is None:
        return "", None
    span = proposition_spans(source)[index]
    left, right = span.start, span.end
    begin = 0
    finish = len(source)
    for boundary in _SENTENCE_BOUNDARY.finditer(source):
        if boundary.end() <= left:
            begin = boundary.end()
        elif boundary.start() >= right:
            finish = boundary.end()
            break
    # Contrast separates unrelated modal/negative clauses. Introductory fiction frames
    # are retained so a model cannot erase their scope by choosing an inner evidence span.
    sentence = source[begin:finish]
    if not _NONFACTUAL_FRAME.search(sentence.strip()):
        for contrast in _CONTRAST.finditer(source, begin, finish):
            if contrast.end() <= left:
                begin = contrast.end()
            elif contrast.start() >= right:
                finish = contrast.start()
                break
    return source[begin:finish].strip(), (left, right)


def _quoted_proposition(source: str, draft: ClaimDraft, offsets: tuple[int, int]) -> bool:
    for quote in re.finditer(r'"[^"\n]+"|“[^”\n]+”|「[^」\n]+」', source):
        if not (quote.start() <= offsets[0] < quote.end() or
                offsets[0] <= quote.start() < offsets[1]):
            continue
        text = quote.group()
        if _contains_value(text, draft.object_value) and (
            _FIRST_PERSON.search(text) or any(marker in text for marker in _CHINESE_FIRST_PERSON)
        ):
            return True
    return False


def _subject_aliases(draft: ClaimDraft) -> set[str]:
    aliases = {normalize_text(draft.subject)}
    simplified = re.sub(r"(?:[-_]?\d+)+$", "", normalize_text(draft.subject))
    aliases.add(simplified.replace("-", " ").replace("_", " "))
    return {alias for alias in aliases if len(alias) >= 3}


def _different_object_subject(normalized_span: str, aliases: set[str]) -> bool:
    if any(alias in normalized_span for alias in aliases):
        return False
    return bool(
        re.match(
            r"^(?:the|a|an)\s+[\w-]+(?:\s+[\w-]+){0,3}\s+"
            r"(?:moved|transferred|shifted|changed|relocated|was|is)\b",
            normalized_span,
        )
    )


def _draft_is_newer(old: Claim, new: ClaimDraft) -> bool:
    if new.valid_start is None:
        return new.assertion_time >= old.assertion_time
    if old.valid_start is None:
        return True
    return new.valid_start >= old.valid_start
