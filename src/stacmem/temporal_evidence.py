"""Conservative temporal evidence accounting, independent of interval parsing."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal

from .source_spans import SourceSpan

DETECTOR = "temporal-evidence-accounting-v1"
EvidenceKind = Literal["date_form", "bare_year", "relative_duration", "period", "modifier"]
Disposition = Literal["parsed_cue", "object_literal", "unresolved"]

_MONTH = (
    r"(?:January|February|March|April|May|June|July|August|September|October|November|December|"
    r"Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)"
)
_EN_UNIT = (
    r"(?:seconds?|minutes?|hours?|days?|weeks?|months?|quarters?|years?|decades?|"
    r"centur(?:y|ies)|semesters?|seasons?|weekends?)"
)
_EN_QUANTITY = (
    r"(?:\d+(?:\.\d+)?|a(?:\s+(?:few|couple(?:\s+of)?))?|an|few|several|some|"
    r"one|two|three|four|five|six|seven|eight|nine|ten)"
)
_ZH_QUANTITY = (
    r"(?:[0-9\u4e00\u4e8c\u4e09\u56db\u4e94\u516d\u4e03\u516b\u4e5d\u5341"
    r"\u767e\u5343\u4e07\u4e24\u51e0\u6570\u534a]+|\u82e5\u5e72)"
)
_ZH_UNIT = (
    r"(?:\u79d2\u949f?|\u5206\u949f|\u5c0f\u65f6|\u5929|\u65e5|\u661f\u671f|"
    r"\u5468|(?:\u4e2a)?\u6708|(?:\u4e2a)?\u5b63\u5ea6|"
    r"(?:\u4e2a)?\u5b66\u671f|(?:\u4e2a)?\u5e74)"
)

# These patterns recognize evidence shapes, not exact intervals. The bare-year
# layer catches unseen qualifiers (e.g. an unknown season/period followed by a year).
_PATTERNS: tuple[tuple[EvidenceKind, re.Pattern[str]], ...] = (
    ("date_form", re.compile(
        r"(?<![A-Za-z0-9_])\d{4}[-/]\d{1,2}(?:[-/]\d{1,2})?"
        r"(?:T\d{2}:\d{2}(?::\d{2})?(?:Z|[+-]\d{2}:\d{2})?)?(?![A-Za-z0-9_])|"
        rf"\b{_MONTH}\s+(?:\d{{1,2}}(?:st|nd|rd|th)?(?:,\s*|\s+)\d{{4}}|"
        r"\d{4}|\d{1,2}(?:st|nd|rd|th)?)\b|"
        rf"\b\d{{1,2}}(?:st|nd|rd|th)?\s+{_MONTH}(?:\s+\d{{4}})?\b|"
        r"\d{4}\u5e74(?:\d{1,2}\u6708(?:\d{1,2}[\u65e5\u53f7])?)?",
        re.I,
    )),
    ("bare_year", re.compile(r"(?<![A-Za-z0-9_])\d{4}(?![A-Za-z0-9_])")),
    ("relative_duration", re.compile(
        rf"\b(?:{_EN_QUANTITY}\s+)?{_EN_UNIT}\s+"
        r"(?:later|earlier|afterwards?|before|after|ago|hence|from\s+now)\b|"
        rf"(?:{_ZH_QUANTITY})?{_ZH_UNIT}(?:\u534a)?(?:\u4e4b|\u4ee5)?[\u524d\u540e]",
        re.I,
    )),
    ("period", re.compile(
        r"\bQ[1-4]\b|\b(?:spring|summer|fall|autumn|winter|quarters?|semesters?)\b|"
        r"\b(?:first|second|latter|former|early|later)\s+half\b|"
        rf"\b(?:beginning|middle|end)\s+of\s+(?:the\s+)?(?:{_EN_UNIT}|\d{{4}})\b|"
        r"\b(?:year|month|quarter)[ -]?(?:end|start)\b|"
        r"[\u5e74\u5b63\u6708][\u521d\u4e2d\u672b\u5e95]|"
        r"[\u4e0a\u4e0b\u524d\u540e]\u534a[\u5e74\u6708]|"
        r"(?:\u7b2c[\u4e00\u4e8c\u4e09\u56db1-4])?\u5b63\u5ea6|"
        r"[\u6625\u590f\u79cb\u51ac][\u5929\u5b63]|"
        r"[\u660e\u53bb\u6b21\u672c]\u5e74",
        re.I,
    )),
    ("modifier", re.compile(
        r"\b(?:eventually|soon|shortly|sometime|someday|recently|previously|formerly|"
        r"later|earlier|subsequently|afterwards?|yesterday|today|tomorrow|"
        r"before|after|since|until|around|approximately|circa)\b|"
        rf"\b(?:this|that|next|previous|last|following|coming|upcoming)\s+{_EN_UNIT}\b|"
        rf"(?:\u672a\u6765|\u8fc7\u53bb|\u63a5\u4e0b\u6765|[\u4e0a\u4e0b\u8fd9\u672c\u6b21]){_ZH_UNIT}|"
        r"\u4e4b\u540e|\u540e\u6765|\u6b21\u65e5|\u7fcc\u65e5|\u6700\u7ec8|"
        r"\u968f\u540e|\u4e0d\u4e45|\u6b21\u6708",
        re.I,
    )),
)


@dataclass(frozen=True)
class TemporalEvidence:
    span: str
    bounds: tuple[int, int]
    kind: EvidenceKind


@dataclass(frozen=True)
class EvidenceAccount:
    evidence: TemporalEvidence
    disposition: Disposition

    def model_dump(self) -> dict[str, Any]:
        return {
            "span": self.evidence.span,
            "codepoint_span": self.evidence.bounds,
            "kind": self.evidence.kind,
            "disposition": self.disposition,
        }


def account_temporal_evidence(
    source: str, span: SourceSpan, *,
    consumed_cues: list[tuple[int, int]], object_value: str,
) -> tuple[EvidenceAccount, ...]:
    """Account for detected evidence by literal containment, never date equality."""
    if source[span.start:span.end] != span.text:
        raise ValueError("Temporal evidence span does not match the original source")
    value = object_value.strip()
    value_pattern = re.escape(value)
    if value and value.isascii() and value[0].isalnum() and value[-1].isalnum():
        value_pattern = rf"(?<!\w){value_pattern}(?!\w)"
    objects = [
        (span.start + match.start(), span.start + match.end())
        for match in re.finditer(value_pattern, span.text, re.I)
    ] if value else []
    evidence_by_bounds = {}
    for kind, pattern in _PATTERNS:
        for match in pattern.finditer(span.text):
            bounds = (span.start + match.start(), span.start + match.end())
            evidence_by_bounds.setdefault(
                bounds, TemporalEvidence(source[slice(*bounds)], bounds, kind),
            )
    accounts = []
    for bounds, evidence in sorted(evidence_by_bounds.items()):
        disposition: Disposition = "unresolved"
        if any(start <= bounds[0] and bounds[1] <= end for start, end in consumed_cues):
            disposition = "parsed_cue"
        elif any(start <= bounds[0] and bounds[1] <= end for start, end in objects):
            disposition = "object_literal"
        accounts.append(EvidenceAccount(evidence, disposition))
    return tuple(accounts)
