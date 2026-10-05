"""UTC millisecond and half-open interval helpers."""

from __future__ import annotations

import datetime as dt
import math
import time
from typing import Any

UTC = dt.UTC
INF_MS = 253402300799999  # 9999-12-31T23:59:59.999Z
NEG_INF_MS = -62135596800000  # 0001-01-01T00:00:00Z


def now_ms() -> int:
    return time.time_ns() // 1_000_000


def ensure_ms(value: int | float | str | dt.datetime | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("boolean is not a timestamp")
    if isinstance(value, (int, float)):
        numeric = float(value)
        return int(numeric if abs(numeric) >= 1e11 else numeric * 1000)
    if isinstance(value, dt.datetime):
        parsed = value
    elif isinstance(value, str):
        raw = value.strip()
        if not raw:
            return None
        if raw.isdigit() or (raw.startswith("-") and raw[1:].isdigit()):
            return ensure_ms(int(raw))
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        parsed = dt.datetime.fromisoformat(raw)
    else:
        raise TypeError(f"unsupported timestamp type: {type(value)!r}")
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return int(parsed.astimezone(UTC).timestamp() * 1000)


def to_iso(value: int | None) -> str | None:
    if value is None:
        return None
    return dt.datetime.fromtimestamp(value / 1000, tz=UTC).isoformat().replace("+00:00", "Z")


def contains(start: int | None, end: int | None, point: int) -> bool:
    return (start is None or start <= point) and (end is None or point < end)


def overlaps(
    a_start: int | None,
    a_end: int | None,
    b_start: int | None,
    b_end: int | None,
) -> bool:
    left = max(a_start if a_start is not None else NEG_INF_MS, b_start or NEG_INF_MS)
    right = min(a_end if a_end is not None else INF_MS, b_end or INF_MS)
    return left < right


def interval_distance_ms(start: int | None, end: int | None, point: int) -> int:
    if contains(start, end, point):
        return 0
    if start is not None and point < start:
        return start - point
    if end is not None and point >= end:
        return point - end
    return 0


def temporal_decay(distance_ms: int, half_life_days: float = 30.0) -> float:
    if distance_ms <= 0:
        return 1.0
    half_life_ms = half_life_days * 86_400_000
    return math.exp(-math.log(2) * distance_ms / half_life_ms)


def coalesce_time(value: Any, fallback: int) -> int:
    parsed = ensure_ms(value)
    return fallback if parsed is None else parsed
