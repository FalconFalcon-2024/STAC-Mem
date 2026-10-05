from __future__ import annotations

import pytest

from stacmem.config import AppConfig
from stacmem.pipeline import StacMemory


@pytest.fixture
def memory(tmp_path):
    config = AppConfig()
    config.runtime.database_path = str(tmp_path / "ledger.sqlite3")
    config.extraction.provider = "rule"
    config.embedding.provider = "hash"
    config.embedding.dimensions = 128
    config.rerank.provider = "none"
    # Exercise the lower-level legacy engine independently of the fixed public state runtime.
    config.admission.mode = "lexical_v1"
    config.temporal_grounding.enabled = False
    config.temporal_grounding.mode = "certificate_v1"
    config.conflict.transition_mode = "lexical_v1"
    config.conflict.temporal_relation_mode = "transaction_fallback_v1"
    service = StacMemory.from_app_config(config, _install_contracts=False)
    yield service
    service.close()
