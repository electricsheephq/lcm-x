"""#634: lifecycle races must not wedge real engine publication across clones."""

import pytest

import hermes_lcm.engine as engine_module
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from tests.test_compression_boundary import _stub_summarizer, _turn
from tests.test_issue_634_lifecycle_cas import CONV, _other_writes_before_next_write


@pytest.mark.parametrize("race", ["finalize", "bind"])
@pytest.mark.parametrize("interleaved", [True, False], ids=["race", "sequential-control"])
def test_lifecycle_race_keeps_the_other_engine_compacting(
    tmp_path, monkeypatch, caplog, race, interleaved,
):
    monkeypatch.setattr(engine_module, "summarize_with_escalation", _stub_summarizer())
    config = LCMConfig(
        database_path=str(tmp_path / "lcm.db"), fresh_tail_count=6,
        leaf_chunk_tokens=400, large_output_externalization_path=str(tmp_path / "ext"),
    )
    proto = LCMEngine(config=config, hermes_home=str(tmp_path / "home"))
    a, b = proto.clone_for_agent(), proto.clone_for_agent()
    engines = [proto, a, b]
    hosts = {"A": [], "B": []}
    turns = {"A": 0, "B": 0}

    def start(engine, sid):
        engine.on_session_start(
            sid, platform="telegram", context_length=200_000, conversation_id=CONV,
        )

    def compact(engine, sid, count):
        for _ in range(count):
            turns[sid] += 1
            hosts[sid].extend(_turn(turns[sid]))
            engine.ingest(hosts[sid])
        engine.compress(list(hosts[sid]), force=True)
        return engine._last_compression_status

    try:
        start(a, "A")
        assert compact(a, "A", 12) == "compacted"

        if race == "finalize":
            def other_write():
                start(b, "B")
                assert compact(b, "B", 12) == "compacted"

            def stale_write():
                a.on_session_end("A", [])
        else:
            # Two clones of the SAME session can end with different in-process
            # frontiers. The older end lands first; the newer end races resume.
            old_frontier = a._last_compacted_store_id
            start(b, "A")
            hosts["B"] = hosts["A"]
            turns["B"] = turns["A"]
            assert compact(b, "B", 6) == "compacted"
            new_frontier = b._last_compacted_store_id
            assert new_frontier > old_frontier
            a.on_session_end("A", [])

            def other_write():
                b.on_session_end("A", [])

            def stale_write():
                a._lifecycle.bind_session("A", conversation_id=CONV)

        real = a._lifecycle._conn
        if interleaved:
            _, fired = _other_writes_before_next_write(a._lifecycle, other_write)
            try:
                stale_write()
                assert fired == [1]
            finally:
                a._lifecycle._conn = real
        else:
            other_write()
            stale_write()

        # Exercise publication before checking row metadata: on main the same
        # engine fails after stale finalize, and resumed A fails after stale bind.
        caplog.clear()
        sid = "B"
        if race == "bind":
            b.shutdown()
            b = proto.clone_for_agent()
            engines.append(b)
            sid = "A"
            start(b, sid)
        frontier = b._last_compacted_store_id
        for restart in range(2):
            assert compact(b, "B", 6) == "compacted", [r.message for r in caplog.records]
            assert b._last_compacted_store_id > frontier
            frontier = b._last_compacted_store_id
            if race == "finalize":
                assert b._store.get_session_count("A") == 24
            assert b._store.get_session_count(sid) == 2 * turns["B"]
            assert not any("publication_invariant_conflict" in r.message for r in caplog.records)
            if restart == 0:
                b.on_session_end(sid, [])
                b.shutdown()
                b = proto.clone_for_agent()
                engines.append(b)
                start(b, sid)
    finally:
        for engine in reversed(engines):
            engine.shutdown()
