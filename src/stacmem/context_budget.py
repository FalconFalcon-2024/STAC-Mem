"""Deterministic multilingual context-size estimation.

This is deliberately an estimate rather than a model-specific tokenizer. ASCII words are
approximated in four-character pieces, while non-ASCII characters and punctuation are counted
conservatively. The method is stable, offline, and visible in evidence diagnostics.
"""

from __future__ import annotations

import math
import re

ASCII_WORD = re.compile(r"[A-Za-z0-9_]+")


def estimate_tokens(text: str) -> int:
    total = 0
    position = 0
    for match in ASCII_WORD.finditer(text):
        total += _non_ascii_and_punctuation(text[position : match.start()])
        total += max(1, math.ceil(len(match.group()) / 4))
        position = match.end()
    return total + _non_ascii_and_punctuation(text[position:])


def _non_ascii_and_punctuation(text: str) -> int:
    return sum(1 for character in text if not character.isspace())
