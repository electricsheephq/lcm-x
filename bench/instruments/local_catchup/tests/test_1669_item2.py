"""Issue #1669: round-five instrument claims, reproduced before implementation."""
from pathlib import Path
import threading

import pytest

from bench.instruments.local_catchup import common, measurement

pytest_plugins = ["bench.instruments.local_catchup.tests.test_corpus_audit"]
WORKFLOW = Path(__file__).resolve().parents[4]/".github/workflows/lcmx-1586-floor.yml"


@pytest.mark.parametrize("foreground,reader", [(False, False), (True, False), (True, True)])
def test_item2_shutdown_join_allowed_but_request_join_denied(tmp_path, monkeypatch, foreground, reader):
    # Stop runtime immediately after it installs the real join wrapper.
    request = tmp_path/"request.json"
    common.dump(request, {**{k: str(tmp_path) for k in
        ("db", "root", "home", "scratch", "out", "lcmx_repo", "hermes_source", "model_cache")}, "state": "ON"})
    monkeypatch.setattr(threading.Thread, "join", threading.Thread.join)
    monkeypatch.setattr(measurement, "isolated", lambda *a: None)
    monkeypatch.setattr(measurement, "product", lambda *a: None)
    monkeypatch.setattr(measurement, "guards", lambda *a: None)
    def end(*a):
        raise ValueError("wrapper installed")
    monkeypatch.setattr(measurement, "config", end)
    with pytest.raises(ValueError, match="wrapper installed"):
        measurement.runtime(request)
    thread = threading.Thread(target=lambda: None, name="lcm-query-embed" if reader else "embedding-backfill")
    thread.start()
    if reader:
        errors = []
        def query_join():
            try:
                with common.foreground_request():
                    thread.join()
            except RuntimeError as exc:
                errors.append(str(exc))
        query = threading.Thread(target=query_join, name="i2-recall")
        query.start()
        query.join()
        assert not errors
        assert not (tmp_path/"events.jsonl").exists()
    elif foreground:
        with common.foreground_request(), pytest.raises(RuntimeError, match="embedding join"):
            thread.join()
        assert common.load(tmp_path/"events.jsonl")["violation"] == "foreground_join"
    else:
        thread.join()
        assert not (tmp_path/"events.jsonl").exists()
