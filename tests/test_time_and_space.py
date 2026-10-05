from __future__ import annotations

from stacmem.models import Place
from stacmem.spatial import haversine_km, spatial_similarity
from stacmem.time_utils import contains, overlaps


def test_half_open_interval_boundaries() -> None:
    assert contains(100, 200, 100)
    assert contains(100, 200, 199)
    assert not contains(100, 200, 200)
    assert not overlaps(100, 200, 200, 300)
    assert overlaps(100, 201, 200, 300)


def test_place_hierarchy_and_coordinates() -> None:
    hangzhou = Place(
        name="Hangzhou",
        hierarchy=["CN", "Zhejiang", "Hangzhou"],
        latitude=30.2741,
        longitude=120.1551,
    )
    xihu = Place(
        name="Xihu",
        hierarchy=["CN", "Zhejiang", "Hangzhou", "Xihu"],
        latitude=30.2590,
        longitude=120.1300,
    )
    tokyo = Place(
        name="Tokyo",
        hierarchy=["JP", "Tokyo"],
        latitude=35.6762,
        longitude=139.6503,
    )
    assert spatial_similarity(hangzhou, xihu) > spatial_similarity(hangzhou, tokyo)
    assert (haversine_km(hangzhou, xihu) or 999) < 10
