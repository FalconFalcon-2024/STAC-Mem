"""Executable SDK wiring example, with explicit hand-authored structures."""

from pathlib import Path
from tempfile import TemporaryDirectory

from stacmem import QueryFrame, StandaloneMemory
from stacmem.standalone_demo import fixture
from stacmem.time_utils import ensure_ms


def main() -> None:
    with (
        TemporaryDirectory(prefix="stacmem-agent-") as folder,
        StandaloneMemory.offline(Path(folder) / "memory.db") as memory,
    ):
        memory.remember(**fixture("user", "s1", "On 2024-08-01 I joined Aster.", "Aster"))
        memory.remember(**fixture(
            "user", "s2", "On 2024-10-15 I left Aster and joined Birch.", "Birch",
            date="2024-10-15",
        ))
        query = "Where do I currently work?"
        evidence = memory.search(owner_id="user", query=query, frame=QueryFrame(
            owner_id="user", raw_query=query, target_subject="user",
            target_predicate="employment.organization", temporal_intent="current",
            query_time=ensure_ms("2025-01-01"),
        ))
        print(evidence["context"])
        assert [item["claim"]["object_value"] for item in evidence["claims"]] == ["Birch"]
        print("SDK integration completed; no model requests.")


if __name__ == "__main__":
    main()
