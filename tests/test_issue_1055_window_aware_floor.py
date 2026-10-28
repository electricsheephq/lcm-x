"""#1055: effective session windows scale only the non-tool ingest floor."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

import hermes_lcm.ingest_protection as protection
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from hermes_lcm.externalize import is_externalized_placeholder
from hermes_lcm.store import MessageStore
from hermes_lcm.tokens import count_tokens


SMALL = 65_536
LARGE = 1_048_576
SESSION = "window-floor"


def _text(n=143_774):
    prose = "This document describes ordinary plans and useful observations. "
    return (prose * (n // len(prose) + 1))[:n]


def _assistant_text(n=143_774):
    # Distinct sentences exercise the floor without triggering the existing loop quarantine.
    prose = " ".join(f"Paragraph {i} describes ordinary plans and useful observations." for i in range(n // 40 + 1))
    return prose[:n]


def _config(tmp_path, **overrides):
    return LCMConfig(**{
        "database_path": str(tmp_path / "lcm.db"),
        "large_output_externalization_enabled": True,
        "large_output_externalization_threshold_chars": 12_000,
        "large_output_active_replay_stubbing_enabled": True,
        "embeddings_enabled": False,
        "summary_model": "stub-model",
        **overrides,
    })


def _engine(tmp_path, window=LARGE, *, config=None, session=SESSION):
    engine = LCMEngine(config=config or _config(tmp_path), hermes_home=str(tmp_path / "hermes"))
    kwargs = {} if window is None else {"context_length": window}
    engine.on_session_start(session, platform="cli", conversation_id=session, **kwargs)
    return engine


@pytest.fixture
def engine_factory(tmp_path):
    engines = []

    def make(window=LARGE, **kwargs):
        engine = _engine(tmp_path, window, **kwargs)
        engines.append(engine)
        return engine

    yield make
    for engine in engines:
        engine._close_storage()


def _rows(engine, session=SESSION):
    return engine._store.get_session_messages(session)


def _assert_unique(engine, expected):
    rows = _rows(engine)
    assert [r["store_id"] for r in rows] == expected
    keys = [engine._message_replay_identity(r, stored_row=True) for r in rows]
    assert len(keys) == len(set(keys))


def test_143774_char_user_message_stays_inline_at_1m(engine_factory):
    engine = engine_factory()
    text = _text()
    assert len(text) == 143_774
    engine.ingest([{"role": "user", "content": text}])
    [row] = engine._store.get_session_messages(SESSION)
    assert row["content"] == text


@pytest.mark.parametrize("window,char_floor,token_floor", [
    (SMALL, 100_000, 25_000), (200_000, 100_000, 25_000),
    (LARGE, 419_432, 104_858), (0, 100_000, 25_000), (None, 100_000, 25_000),
])
def test_exact_char_and_token_boundaries(tmp_path, monkeypatch, window, char_floor, token_floor):
    config = _config(tmp_path)
    monkeypatch.setattr(protection, "count_tokens", lambda _: 0)
    for length in (char_floor - 1, char_floor, char_floor + 1):
        assert protection._non_tool_text_over_floor(_text(length), config, context_window_tokens=window) == (
            length > char_floor
        )
    for tokens in (token_floor - 1, token_floor, token_floor + 1):
        monkeypatch.setattr(protection, "count_tokens", lambda _, tokens=tokens: tokens)
        assert protection._non_tool_text_over_floor(_text(12_001), config, context_window_tokens=window) == (
            tokens >= token_floor
        )


def test_token_count_guards_stay_unchanged(tmp_path, monkeypatch):
    def forbidden_count(_):
        pytest.fail("token counter should not run")

    monkeypatch.setattr(protection, "count_tokens", forbidden_count)
    assert not protection._non_tool_text_over_floor(_text(12_000), _config(tmp_path))
    assert not protection._non_tool_text_over_floor(
        _text(), _config(tmp_path, large_output_externalization_enabled=False), context_window_tokens=LARGE,
    )


def test_dense_script_reaches_token_floor_before_char_floor(tmp_path):
    text = "漢字文章測試" * 7_000
    assert 12_000 < len(text) < 100_000
    assert 25_000 <= count_tokens(text) < 104_858
    config = _config(tmp_path)
    assert protection._non_tool_text_over_floor(text, config, context_window_tokens=SMALL)
    assert not protection._non_tool_text_over_floor(text, config, context_window_tokens=LARGE)


@pytest.mark.parametrize("role", ["user", "assistant"])
@pytest.mark.parametrize("window,externalized", [(SMALL, True), (200_000, True), (LARGE, False), (None, True)])
def test_ordinary_text_uses_bound_window(engine_factory, role, window, externalized):
    engine = engine_factory(window)
    text = _assistant_text() if role == "assistant" else _text()
    engine.ingest([{"role": role, "content": text}])
    [row] = _rows(engine)
    assert is_externalized_placeholder(row["content"]) == externalized
    if externalized:
        assert row["content"].startswith(f"[Externalized payload: kind=raw_payload; role={role};")
        assert "read it with lcm_expand" in row["content"]
    else:
        assert row["content"] == text


@pytest.mark.parametrize("before,after", [(SMALL, LARGE), (LARGE, SMALL), (None, LARGE)])
@pytest.mark.parametrize("fresh_engine", [False, True], ids=["same-engine", "fresh-engine-same-db"])
def test_rebind_replay_keeps_ids_and_content_keys(engine_factory, before, after, fresh_engine):
    engine = engine_factory(before)
    history = [
        {"role": "user", "content": "opening question"},
        {"role": "assistant", "content": "opening answer"},
        {"role": "user", "content": _text()},
        {"role": "assistant", "content": "document received"},
    ]
    engine.ingest([dict(m) for m in history])
    stored = _rows(engine)
    ids = [r["store_id"] for r in stored]
    assert is_externalized_placeholder(stored[2]["content"]) == (before != LARGE)
    if fresh_engine:
        engine._close_storage()
        engine = engine_factory(after)
    else:
        engine.on_session_start(SESSION, platform="cli", conversation_id=SESSION, context_length=after)
    assert engine._store.get_context_window_tokens(SESSION) == after
    for _ in range(2):
        engine.ingest([dict(m) for m in history])
        _assert_unique(engine, ids)
        assert [r["content"] for r in _rows(engine)] == [r["content"] for r in stored]
    assert engine._session_end_store_prefix_count(SESSION, history) == len(history)
    engine.on_session_end(SESSION, [dict(m) for m in history])
    _assert_unique(engine, ids)
    suffix = {"role": "user", "content": "NEW SUFFIX " + _text()}
    engine.on_session_start(SESSION, platform="cli", conversation_id=SESSION, context_length=after)
    engine.ingest([dict(m) for m in history] + [suffix])
    assert len(_rows(engine)) == len(ids) + 1
    assert is_externalized_placeholder(_rows(engine)[-1]["content"]) == (after == SMALL)


@pytest.mark.parametrize("window,externalized", [(SMALL, True), (LARGE, False), (None, True)])
def test_store_single_and_batch_use_session_window(tmp_path, window, externalized):
    store = MessageStore(tmp_path / "lcm.db", ingest_protection_config=_config(tmp_path),
                         hermes_home=str(tmp_path / "hermes"))
    try:
        if window is not None:
            store.set_context_window_tokens(SESSION, window)
        store.append(SESSION, {"role": "user", "content": _text()})
        store.append_batch(SESSION, [{"role": "assistant", "content": _assistant_text()}])
        assert all(is_externalized_placeholder(r["content"]) == externalized
                   for r in store.get_session_messages(SESSION))
        store.append("unknown", {"role": "user", "content": _text()})
        assert is_externalized_placeholder(store.get_session_messages("unknown")[0]["content"])
    finally:
        store.close()


@pytest.mark.parametrize("old_window,new_window", [(SMALL, LARGE), (LARGE, SMALL)])
def test_late_session_end_suffix_uses_target_session_window(engine_factory, old_window, new_window):
    engine = engine_factory(old_window)
    history = [{"role": "user", "content": "original session opening"}]
    engine.ingest(history)
    old_id = _rows(engine)[0]["store_id"]
    engine.on_session_start("foreground", platform="cli", conversation_id="foreground", context_length=new_window)
    engine.on_session_end(SESSION, history + [{"role": "assistant", "content": _assistant_text()}])
    rows = _rows(engine)
    assert len(rows) == 2 and rows[0]["store_id"] == old_id
    assert is_externalized_placeholder(rows[-1]["content"]) == (old_window == SMALL)
    engine.on_session_end(SESSION, history + [{"role": "assistant", "content": _assistant_text()}])
    assert len(_rows(engine)) == 2
    assert engine._session_id == "foreground"
    assert _rows(engine, "foreground") == []


@pytest.mark.parametrize("before,after", [(SMALL, LARGE), (LARGE, SMALL), (LARGE, LARGE)])
def test_host_whitespace_rewrite_reuse_and_non_reuse(engine_factory, before, after):
    engine = engine_factory(before)
    message = {"role": "user", "content": _text() + "\n"}
    history = [{"role": "user", "content": "opening question"},
               {"role": "assistant", "content": "opening answer"}, message]
    engine.ingest(history)
    stored = _rows(engine)[-1]
    ids = [r["store_id"] for r in _rows(engine)]
    engine.update_model("fixture-model", after)
    message["content"] = message["content"].strip()  # actual host in-place edge rewrite
    engine.ingest(history)
    _assert_unique(engine, ids)
    assert _rows(engine)[-1]["content"] == stored["content"]
    override = engine._store.read_metadata_json(f"host_rewrite_identity:{stored['store_id']}")
    assert override is not None
    assert is_externalized_placeholder(override["content"]) == (before == SMALL or after == SMALL)
    if before == SMALL:
        assert override["content"] == stored["content"] and override["strip_payload"] is True
    # Fresh-process override decoding is separate from transcript admission; grow/shrink
    # admission with unchanged raw transcripts is exercised end to end above.
    fresh = engine_factory(after)
    assert fresh._message_replay_identity(stored, stored_row=True, with_host_rewrite=True) == (
        fresh._message_replay_identity(message)
    )
    _assert_unique(fresh, ids)


@pytest.mark.parametrize("metadata", [
    {}, {"context_length": 0}, {"context_length": None}, {"context_length": "invalid"},
    {"context_length": SMALL}, {"context_length": LARGE, "model": "stale-model"},
])
def test_authoritative_update_model_publishes_after_metadata_early_returns(engine_factory, metadata):
    engine = engine_factory(None)
    engine.update_model("fixture-model", LARGE)
    engine.on_session_start("next-session", platform="cli", conversation_id="next-session", **metadata)
    assert engine.context_length == LARGE
    assert engine._store.get_context_window_tokens("next-session") == LARGE
    engine._store.append("next-session", {"role": "user", "content": _text()})
    assert _rows(engine, "next-session")[0]["content"] == _text()


def test_effective_provider_cap_and_clear_are_mirrored(engine_factory):
    engine = engine_factory(None)
    engine.update_model("gpt-5.5", LARGE, provider="openai-codex")
    engine.on_session_start(SESSION, platform="cli", conversation_id=SESSION, context_length=LARGE, model="gpt-5.5")
    assert engine.raw_context_length == LARGE and engine.context_length == 272_000
    assert engine._store.get_context_window_tokens(SESSION) == 272_000
    engine.ingest([{"role": "user", "content": _text()}])
    assert is_externalized_placeholder(_rows(engine)[0]["content"])
    engine.update_model("fixture-model", 0)
    assert engine._store.get_context_window_tokens(SESSION) == 272_000  # until the session's next protection
    engine.ingest([{"role": "user", "content": _text()}, {"role": "assistant", "content": "ok"}])
    assert engine._store.get_context_window_tokens(SESSION) == 0


def test_update_model_before_the_next_session_keeps_the_previous_sessions_window(engine_factory):
    engine = engine_factory(LARGE)
    history = [{"role": "user", "content": "original session opening"}]
    engine.ingest(history)
    engine.update_model("next-model", SMALL)  # Hermes resolves the next session's model before binding it
    assert engine._store.get_context_window_tokens(SESSION) == LARGE
    engine.on_session_start("next", platform="cli", conversation_id="next", context_length=SMALL, model="next-model")
    assert engine._store.get_context_window_tokens("next") == SMALL
    engine.on_session_end(SESSION, history + [{"role": "assistant", "content": _assistant_text()}])
    assert len(_rows(engine)) == 2 and not is_externalized_placeholder(_rows(engine)[-1]["content"])


def test_in_session_model_change_applies_to_the_next_protection(engine_factory):
    engine = engine_factory(LARGE)
    history = [{"role": "user", "content": "opening"}]
    engine.ingest(history)
    engine.update_model("smaller-model", SMALL)
    engine.ingest(history + [{"role": "assistant", "content": _assistant_text()}])
    assert is_externalized_placeholder(_rows(engine)[-1]["content"])
    assert engine._store.get_context_window_tokens(SESSION) == SMALL


def test_clone_and_shared_config_window_isolation(tmp_path, engine_factory):
    config = _config(tmp_path)
    parent = engine_factory(LARGE, config=config)
    clone = parent.clone_for_agent()
    sibling = engine_factory(SMALL, config=config, session="sibling")
    try:
        clone.on_session_start("clone", platform="cli", conversation_id="clone", context_length=SMALL)
        for engine, session, externalized in [(parent, SESSION, False), (clone, "clone", True), (sibling, "sibling", True)]:
            engine.ingest([{"role": "user", "content": _text()}])
            assert is_externalized_placeholder(_rows(engine, session)[0]["content"]) == externalized
        assert parent._store.get_context_window_tokens(SESSION) == LARGE
        assert parent._store.get_context_window_tokens("clone") == 0
        assert clone._store.get_context_window_tokens(SESSION) == 0
        assert not any("context_window" in k for k in vars(config))
    finally:
        clone._close_storage()


def test_storage_rebind_publishes_window_without_profile_leakage(tmp_path, engine_factory):
    engine = engine_factory(LARGE, config=_config(tmp_path, database_path=""))
    new_home = tmp_path / "other-profile"
    engine.on_session_start("new-profile", platform="cli", conversation_id="new-profile", hermes_home=str(new_home))
    assert engine._store.db_path == new_home / "lcm.db"
    assert engine._store.get_context_window_tokens(SESSION) == 0
    assert engine._store.get_context_window_tokens("new-profile") == LARGE
    engine.ingest([{"role": "user", "content": _text()}])
    assert _rows(engine, "new-profile")[0]["content"] == _text()


def test_offline_import_defaults_to_original_floor(tmp_path):
    path = Path(__file__).resolve().parents[1] / "scripts" / "import_lossless_claw.py"
    spec = importlib.util.spec_from_file_location("issue1055_importer", path)
    importer = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = importer
    spec.loader.exec_module(importer)
    # Run the actual import storage boundary against an isolated store/fixture.
    store = MessageStore(tmp_path / "lcm.db", ingest_protection_config=_config(tmp_path))
    try:
        importer._ensure_import_table(store.connection)
        candidate = importer.ImportCandidate(
            source_message_id=1, source_conversation_id=1, target_session_id=SESSION,
            source_message_key="fixture:1", source_session=SESSION, source="fixture",
            role="user", content=_text(), tool_call_id=None, tool_calls=None,
            tool_name=None, timestamp=1.0, token_estimate=0,
        )
        importer._insert_import_candidate(store.connection, import_id="fixture", candidate=candidate,
                                          protection_config=_config(tmp_path), target_path=store.db_path)
        assert is_externalized_placeholder(store.get_session_messages(SESSION)[0]["content"])
    finally:
        store.close()


@pytest.mark.parametrize("window", [SMALL, LARGE])
def test_tool_and_media_thresholds_and_stub_format_do_not_scale(tmp_path, window):
    config = _config(tmp_path)
    tool = protection.protect_message_for_ingest(
        {"role": "tool", "content": _text(12_001), "tool_call_id": "call-fixture", "tool_name": "fixture"},
        config, str(tmp_path / "hermes"), SESSION, context_window_tokens=window,
    )
    assert tool["content"].startswith("[Externalized tool output:")
    assert is_externalized_placeholder(tool["content"])
    media = protection.protect_message_for_ingest(
        {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64," + "A" * 20_000}}]},
        config, str(tmp_path / "hermes"), SESSION, context_window_tokens=window,
    )
    assert "Externalized" in json.dumps(media["content"])
    assert "A" * 100 not in json.dumps(media["content"])
