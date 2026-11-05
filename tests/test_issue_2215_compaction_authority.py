"""#2215: LCMEngine declares compress() as its only change to the host transcript (compacts_outside_compress)."""

import pytest

from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine


@pytest.fixture
def engines(tmp_path):
    engine = LCMEngine(config=LCMConfig(database_path=str(tmp_path / "lcm.db"), embeddings_enabled=False))
    clone = engine.clone_for_agent()
    yield engine, clone
    clone.shutdown()
    engine.shutdown()


def test_declaration_is_the_false_singleton_on_the_class_the_engine_and_its_clone(engines):
    assert LCMEngine.__dict__["compacts_outside_compress"] is False
    for engine in engines:
        assert type(engine) is LCMEngine
        assert engine.compacts_outside_compress is False


def test_host_resolver_reads_compress_as_the_only_authority(engines):
    context_engine = pytest.importorskip("agent.context_engine")
    resolver = getattr(context_engine, "engine_compacts_outside_compress", None)
    if resolver is None:
        pytest.skip("this host has no compaction-authority resolver (hermes-agent#94983)")
    for engine in engines:
        unsafe, reason = resolver(engine)
        assert unsafe is False, reason
