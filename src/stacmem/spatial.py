"""Hierarchical and coordinate-aware place matching."""

from __future__ import annotations

import math

from .models import Place, normalize_text

EARTH_RADIUS_KM = 6371.0088


def haversine_km(a: Place, b: Place) -> float | None:
    if None in (a.latitude, a.longitude, b.latitude, b.longitude):
        return None
    lat1, lon1 = math.radians(a.latitude), math.radians(a.longitude)  # type: ignore[arg-type]
    lat2, lon2 = math.radians(b.latitude), math.radians(b.longitude)  # type: ignore[arg-type]
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(min(1.0, math.sqrt(h)))


def hierarchy_similarity(a: Place, b: Place) -> float | None:
    left = [normalize_text(item) for item in a.hierarchy if item.strip()]
    right = [normalize_text(item) for item in b.hierarchy if item.strip()]
    if not left or not right:
        return None
    common = 0
    for x, y in zip(left, right, strict=False):
        if x != y:
            break
        common += 1
    if common == 0:
        return 0.0
    return common / max(len(left), len(right))


def spatial_similarity(a: Place | None, b: Place | None) -> float:
    if b is None:
        return 1.0
    if a is None:
        return 0.35
    if a.place_id and b.place_id:
        return 1.0 if normalize_text(a.place_id) == normalize_text(b.place_id) else 0.0
    a_names = {normalize_text(item) for item in [a.name, *a.aliases] if item}
    b_names = {normalize_text(item) for item in [b.name, *b.aliases] if item}
    if a_names & b_names:
        return 1.0
    hierarchy = hierarchy_similarity(a, b)
    distance = haversine_km(a, b)
    distance_score: float | None = None
    if distance is not None:
        scale = max(a.radius_km or 25.0, b.radius_km or 25.0, 1.0)
        distance_score = math.exp(-distance / scale)
    available = [score for score in (hierarchy, distance_score) if score is not None]
    return max(available, default=0.0)


def places_disjoint(a: Place | None, b: Place | None) -> bool:
    if a is None or b is None:
        return False
    left = [normalize_text(item) for item in a.hierarchy if item.strip()]
    right = [normalize_text(item) for item in b.hierarchy if item.strip()]
    if left and right:
        # A country/region and one of its descendants overlap. Two sibling
        # cities do not, even though their longest common prefix is the country.
        prefix = left[: len(right)] == right or right[: len(left)] == left
        if not prefix:
            return True
    return spatial_similarity(a, b) < 0.15
