"""Literal proposition boundaries shared by temporal and action grounding."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

# A repeated subject or date introduces an independent coordinated proposition.
# Shared departure/arrival dates are bound explicitly by the temporal grounder.
PROPOSITION_BOUNDARY = re.compile(
    r"[.;!?。！？；,，\n]+|\b(?:but|yet|however|instead)\b|但是|然而|而是|不过|"
    r"\band\s+(?=(?:then\s+)?(?:i|we|he|she|they|my|our)\b|"
    r"(?:on|since|from|before|until|between)\s+\d{4}-\d{2}-\d{2}\b)|"
    r"\band\s+(?:then\s+)?(?=(?:move(?:d)?|join(?:ed)?|left|leave|relocated|"
    r"start(?:ed)?|became|work(?:ed)?|live(?:d)?|prefer)\b)|"
    r"\band\s+(?=(?:later|earlier|subsequently)\s+(?:(?:i|we)\s+)?"
    r"(?:moved|joined|left|relocated|started|became)\b)|"
    r"\bthen\s+(?=(?:(?:i|we)\s+)?(?:on\s+\d{4}-\d{2}-\d{2}|"
    r"moved|joined|left|relocated|started|became|worked|lived|at|in)\b)|"
    r"\b(?:before|after|prior\s+to)\s+(?=(?:moving|joining|leaving|relocating|"
    r"starting|having\s+(?:moved|joined|left))\b)|"
    r"\band\s+(?:then\s+)?(?=(?:at|in)\s+\S)|"
    r"(?:并且|然后|而且)(?=我|我们|\d{4}-\d{2}-\d{2})",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class SourceSpan:
    text: str
    start: int
    end: int


def proposition_spans(source: str) -> list[SourceSpan]:
    spans = []
    start = 0
    for boundary in [*PROPOSITION_BOUNDARY.finditer(source), None]:
        end = boundary.start() if boundary else len(source)
        raw = source[start:end]
        text = raw.strip(" \t:-")
        if text:
            left = start + len(raw) - len(raw.lstrip(" \t:-"))
            spans.append(SourceSpan(text, left, left + len(text)))
        start = boundary.end() if boundary else end
    return spans


def proposition_link(
    source: str, left: SourceSpan, right: SourceSpan
) -> Literal["coordination", "sequence", "subordinate", "clause", "barrier"]:
    """Classify the literal connector without inventing a dependency on distance."""
    gap = source[left.end:right.start]
    if re.search(
        r"[.;!?。！？；\n]|\b(?:but|yet|however|instead)\b|但是|然而|而是|不过", gap, re.I
    ):
        return "barrier"
    if re.search(r"\b(?:before|after|prior\s+to)\b", gap, re.I):
        return "subordinate"
    if re.search(r"\bthen\b|然后", gap, re.I):
        return "sequence"
    if re.search(r"\band\b|并且|而且", gap, re.I):
        return "coordination"
    return "clause"
