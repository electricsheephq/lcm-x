"""I1/I2: a pass cannot adopt state for a later binding with the same IDs."""

import threading
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy

import hermes_lcm.engine as lcm_engine
from tests.test_host_uid_shadow import _state_db
from tests.test_issue_2851_rebind_cursor import WAIT, _bind, _stored, _view
from tests.test_issue_581_582_store_complete_survival import _engine


def test_compaction_cannot_finish_into_a_revisited_binding(tmp_path, monkeypatch):
    _state_db(tmp_path, [("S", None, None), ("B", None, None)])
    engine = _engine(tmp_path)
    engine._hermes_home = engine._store._hermes_home = str(tmp_path)
    summarizing, resume = threading.Event(), threading.Event()
    rows = _view(7)

    def summarize(**kwargs):
        summarizing.set()
        assert resume.wait(WAIT), "summary did not resume"
        return "Earlier turns.\nExpand for details about: turns", 1

    monkeypatch.setattr(lcm_engine, "summarize_with_escalation", summarize)
    try:
        engine.ingest(rows)
        original_rows = _stored(engine, "S")
        with ThreadPoolExecutor(max_workers=1) as pool:
            running = pool.submit(engine.compress, rows, 24_000)
            try:
                assert summarizing.wait(WAIT)
                _bind(engine, "B", "convB")
                _bind(engine, "S", "conv")
                frontier = deepcopy(engine._frontier)
            finally:
                resume.set()
            returned = running.result(timeout=WAIT)
        assert engine._frontier == frontier
        assert returned is rows
        engine.ingest(rows)
        assert _stored(engine, "S") == original_rows
    finally:
        engine.shutdown()
