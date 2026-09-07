"""Native recovery must not convert a rejected LCM claim into committed coverage."""
import copy
from types import SimpleNamespace

import pytest

import hermes_lcm.engine as engine_module
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine


@pytest.fixture
def candidate(tmp_path, monkeypatch):
    cfg = LCMConfig(
        database_path=str(tmp_path / "lcm.db"), native_recovery=True,
        fresh_tail_count=2, fresh_tail_max_tokens=2000, leaf_chunk_tokens=400,
        dynamic_leaf_chunk_enabled=False, embeddings_enabled=False,
        temporal_rollups_enabled=False, empty_lifecycle_gc_enabled=False,
    )
    e = LCMEngine(config=cfg, hermes_home=str(tmp_path))
    e.on_session_start("retained", conversation_id="conversation")
    e.model, e.provider, e.api_mode = "gpt-6-astra", "openai-codex", "codex_responses"
    e.context_length, e.threshold_tokens = 272000, 204000
    e._compression_cancelled_check = lambda: False
    monkeypatch.setattr(engine_module, "summarize_with_escalation", lambda **kw: ("leaf summary", 1))
    yield e
    e.shutdown()


def install_native(monkeypatch, behavior=None):
    calls = []
    class Native:
        _last_compress_aborted = False
        _last_summary_fallback_used = False
        _last_compression_made_progress = True

        def __init__(self, **kwargs):
            assert kwargs["abort_on_summary_failure"] is True
            assert kwargs["model"] == "gpt-6-astra"
            assert kwargs["provider"] == "openai-codex"
            calls.append(self)

        def on_session_start(self, sid):
            assert sid == "retained"

        def compress(self, messages, **kwargs):
            assert callable(self._compression_cancelled_check)
            if behavior:
                return behavior(self, messages)
            return [{"role": "assistant", "content": "native summary"}] + messages[-2:]

    # Tests run with the repository's minimal Hermes dependency stub.
    import sys
    monkeypatch.setitem(sys.modules, "agent.context_compressor", SimpleNamespace(ContextCompressor=Native))
    return calls


def history():
    return [{"role": "user" if i % 2 == 0 else "assistant",
             "content": f"turn {i}: " + "preserved source " * 70} for i in range(12)]


def test_native_recovery_preserves_sources_and_tools(candidate, monkeypatch):
    calls = install_native(monkeypatch)
    def reject_lcm_summary(**kwargs):
        pytest.fail("native recovery must not invoke the LCM summarizer")
    monkeypatch.setattr(engine_module, "summarize_with_escalation", reject_lcm_summary)
    # Stored old history is absent from the active replay. The normal engine
    # cannot prove a new leaf covers those rows. Recovery leaves its frontier intact.
    candidate.ingest([{"role": "user", "content": "old source remains searchable"}])
    messages = history()
    candidate.ingest(messages)
    before = candidate._store._conn.execute("SELECT * FROM messages ORDER BY store_id").fetchall()
    frontier = candidate._lifecycle.get_by_conversation("conversation").current_frontier_store_id
    result = candidate.compress(messages, current_tokens=250000, force=True)
    assert len(calls) == 1
    assert len(result) < len(messages)
    assert result[-2:] == messages[-2:]
    assert candidate.last_compression_status == "host_native"
    assert candidate._lifecycle.get_by_conversation("conversation").current_frontier_store_id == frontier
    assert candidate._store._conn.execute("SELECT * FROM messages ORDER BY store_id").fetchall() == before
    assert candidate._dag._conn.execute("SELECT COUNT(*) FROM summary_nodes").fetchone()[0] == 0
    assert any(s.get("name", s.get("function", {}).get("name")) == "lcm_recall" for s in candidate.get_tool_schemas())


@pytest.mark.parametrize("kind", ["exception", "aborted", "placeholder", "empty", "grows", "cancelled", "superseded", "prune_only", "digest_failure"])
def test_failed_recovery_preserves_exact_input(candidate, monkeypatch, kind):
    original = history()
    def behavior(native, messages):
        if kind == "exception":
            messages[0]["content"] = "mutation stays on copy"
            raise RuntimeError("provider failed")
        if kind == "prune_only":
            native._last_compression_made_progress = False
        if kind == "digest_failure":
            return [{"role": "assistant", "content": "[digest unavailable for segment 1/1 — recover via session_search]"}]
        if kind == "aborted":
            native._last_compress_aborted = True
        if kind == "placeholder":
            native._last_summary_fallback_used = True
        if kind == "empty":
            return []
        if kind == "grows":
            return messages + messages
        if kind == "cancelled":
            cancelled[0] = True
        if kind == "superseded":
            candidate._compression_cancelled_check = lambda: False
        return [{"role": "assistant", "content": "native summary"}] + messages[-2:]
    cancelled = [False]
    candidate._compression_cancelled_check = lambda: cancelled[0]
    install_native(monkeypatch, behavior)
    snapshot = copy.deepcopy(original)
    result = candidate._compress_native_recovery(original)
    assert result is original
    assert original == snapshot
    assert candidate._last_compress_aborted is True


@pytest.mark.parametrize("mode", ["disabled", "unfenced", "cancelled"])
def test_no_native_dispatch_without_admission(candidate, monkeypatch, mode):
    calls = install_native(monkeypatch)
    if mode == "disabled":
        candidate._config.native_recovery = False
    elif mode == "unfenced":
        candidate._compression_cancelled_check = None
    else:
        candidate._compression_cancelled_check = lambda: True
    messages = history()
    assert candidate._compress_native_recovery(messages) is messages
    assert not calls


def test_config_opt_in(monkeypatch):
    monkeypatch.delenv("LCM_NATIVE_RECOVERY", raising=False)
    assert LCMConfig.from_env().native_recovery is False
    monkeypatch.setenv("LCM_NATIVE_RECOVERY", "true")
    assert LCMConfig.from_env().native_recovery is True
