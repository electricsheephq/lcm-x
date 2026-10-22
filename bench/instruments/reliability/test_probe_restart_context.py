"""Restart restoration may invoke observer hooks before the first run_turn initializes its context."""
import ast
import inspect
from types import SimpleNamespace

import pytest

from bench.instruments.reliability import probe
from bench.instruments.reliability.scorers import multiset


@pytest.mark.parametrize("seam", ["engine_ingress", "host_merge"])
def test_restart_observers_preserve_pre_turn_calls_without_inventing_lineage(seam):
    # Execute main's actual context initialization and hook registrations, using synthetic host seams.
    # This includes the callbacks themselves; duplicating their expressions would miss this regression.
    main = ast.parse(inspect.getsource(probe.main)).body[0]
    setup = []
    for node in main.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "cur" for t in node.targets):
            setup.append(node)
        call = getattr(node, "value", None)
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id in (
                "trace_occurrences", "trace_user_merges"):
            setup.append(node)
    assert len(setup) == 3

    class Engine:
        _session_id = "S0"

        def compress(self, messages):
            return messages

    merged = [{"role": "user", "content": "alpha\n\nbeta"}]
    original_calls, events = [], []

    def merge(messages):
        original_calls.append(messages)
        return merged, 1

    helpers = SimpleNamespace(_merge_consecutive_users=merge, _SEQUENCE_REPAIR_PASSES=(merge,))
    scope = {"etype": Engine, "helpers": helpers, "event": lambda **e: events.append(e), "out": {},
             "trace_occurrences": probe.trace_occurrences, "trace_user_merges": probe.trace_user_merges}
    exec(compile(ast.Module(body=setup, type_ignores=[]), "probe-restart-setup", "exec"), scope)
    assert scope["cur"]["turn"] == 0 and "session" not in scope["cur"]
    rows = [{"role": "user", "content": "alpha"}, {"role": "user", "content": "beta"}]
    if seam == "engine_ingress":
        assert Engine().compress(rows) is rows
        assert events[0]["session"] == "S0"  # the observed engine's actual session
    else:
        result, count = helpers._SEQUENCE_REPAIR_PASSES[0](rows)
        assert result is merged and count == 1 and original_calls == [rows]
        assert events[0]["session"] is None  # no turn owner has been observed yet
        host = {"chat": {"keys": {}, "tags": {}}}
        multiset.merge_witnesses(host, events, lambda _s: "chat")
        assert host["chat"]["keys"] == {}
    assert events[0]["event"] == seam
    assert events[0]["turn"] == 0 and events[0]["session_prefix"] == "T"
