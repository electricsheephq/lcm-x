"""I1/I2/I8: complete-list salvage positions govern compaction proofs (#1529)."""

from collections.abc import Callable, Iterator
from copy import deepcopy
import hashlib
from pathlib import Path
from typing import Any

import pytest

import hermes_lcm.engine as engine_module
from tests.harness import make_engine
from tests.test_compression_boundary import _stub_summarizer, _turn
from tests.test_issue_671_stub_first_exit import tool_pair

CLEARED = "[Old tool output cleared to save context space]"
CLEARED_SHA256 = "e4502a9f9608682403df029b95b963304004297fab1897f792ed24a1389b2129"
LONG = "synthetic long lookup output " * 20
Row = dict[str, Any]
Build = Callable[..., Any]


@pytest.fixture
def build(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Build]:
    engines: list[Any] = []
    factory: Any = make_engine
    summarizer: Any = _stub_summarizer
    monkeypatch.setattr(engine_module, "summarize_with_escalation", summarizer())

    def create(*, restart: bool = False) -> Any:
        folder = tmp_path / "session"
        folder.mkdir(exist_ok=True)
        engine = factory(
            folder, window=200_000, session_id="S", conversation_id="conversation",
            fresh_tail_count=9, leaf_chunk_tokens=400,
            large_output_externalization_enabled=False,
            large_output_active_replay_stubbing_enabled=False,
            temporal_rollups_enabled=False, embeddings_enabled=False,
            deferred_maintenance_enabled=False,
        )
        engines.append(engine)
        if restart:
            assert engine._compress_commit_proof is None
            assert engine._durable_commit_proof_payload() is not None
        return engine

    yield create
    for engine in engines:
        engine.shutdown()


def fixture(*, uids: bool = False) -> list[Row]:
    turn: Any = _turn
    pair: Any = tool_pair
    rows: list[Row] = [row for i in range(12) for row in turn(i)]
    for cid, text, stamp in (("older", "older lookup", 8), ("middle", "middle lookup", 9),
                             ("c", "repeat", 10)):
        rows += [{"role": "user", "content": text, "timestamp": stamp},
                 *pair(cid, LONG + cid, tool_name="lookup")]
    if uids:
        for index, row in enumerate(rows):
            row["message_uid"] = f"{index + 1:032x}"
    return rows


def compact(build: Build, *, mode: str = "warm", uids: bool = False) -> tuple[Any, list[Row]]:
    engine = build()
    original = fixture(uids=uids)
    engine.ingest(deepcopy(original))
    output: list[Row] = engine.compress(deepcopy(original), force=True)
    assert engine._last_compression_status == "compacted"
    assert engine._last_compacted_store_id > 0
    assert output[-3:] == original[-3:]
    sid = engine._session_id
    engine.on_session_end(sid, deepcopy(original))
    engine.on_session_start(
        "rotated" if mode == "rotating" else sid,
        boundary_reason="compression", old_session_id=sid,
        conversation_id=engine._conversation_id, platform="cli",
    )
    if mode == "cold":
        engine.shutdown()
        engine = build(restart=True)
    elif mode == "warm":
        assert engine._compress_commit_proof and not engine._compress_commit_proof.get("consulted")
        assert engine._ingest_cursor == len(output)
    return engine, output


def stored(engine: Any) -> list[Row]:
    rows: list[Row] = engine._store.get_session_messages(engine._session_id)
    return rows


def assert_retry(engine: Any, host: list[Row], before: list[Row], content: str) -> None:
    engine.ingest(host)
    after = stored(engine)
    assert after[:len(before)] == before, "existing durable rows changed"
    added = after[len(before):]
    counts = {
        "user": sum(row["role"] == "user" and row["content"] == "repeat" for row in added),
        "call": sum(any(call["id"] == "c" for call in row.get("tool_calls") or []) for row in added),
        "result": sum(row["role"] == "tool" and row.get("tool_call_id") == "c"
                      and row["content"] == content for row in added),
    }
    assert counts == {"user": 1, "call": 1, "result": 1}
    if content == CLEARED:
        assert any(hashlib.sha256(row["content"].encode()).hexdigest() == CLEARED_SHA256 for row in added)
    call_index = next(i for i, row in enumerate(added) if any(
        call["id"] == "c" for call in row.get("tool_calls") or []))
    assert added[call_index + 1]["tool_call_id"] == "c"


@pytest.mark.parametrize("mode", ["warm", "cold", "rotating"])
@pytest.mark.parametrize("uids", [False, True], ids=["unstamped-tools", "message-uids"])
@pytest.mark.parametrize("content", [CLEARED, "genuinely different retry output"], ids=["placeholder", "changed-text"])
def test_retry_preserves_every_role(build: Build, mode: str, uids: bool, content: str) -> None:
    engine, host = compact(build, mode=mode, uids=uids)
    before = stored(engine)
    host[-3]["timestamp"] = 20
    host[-1]["content"] = content
    if uids:
        for index, row in enumerate(host[-3:]):
            row["message_uid"] = f"{100 + index:032x}"
    assert_retry(engine, host, before, content)


@pytest.mark.parametrize("mode", ["warm", "cold"])
@pytest.mark.parametrize("newer", [0, 1, 2], ids=["newest", "second-newest", "older-than-prefix"])
def test_salvage_guard_uses_complete_list(build: Build, mode: str, newer: int) -> None:
    engine, host = compact(build, mode=mode)
    before = stored(engine)
    host[-1]["content"] = CLEARED
    pair: Any = tool_pair
    for index in range(newer):
        host.extend(pair(f"new-{index}", "new result", tool_name="lookup"))
    engine.ingest(host)
    added = stored(engine)[len(before):]
    assert stored(engine)[:len(before)] == before
    # The prefix ends at c, but two results appended AFTER it permit true salvage.
    target = [row for row in added if row.get("tool_call_id") == "c"]
    assert len(target) == (1 if newer < 2 else 0)
    if target:
        assert target[0]["content"] == CLEARED
    else:
        assert len(added) == 2 * newer


@pytest.mark.parametrize("content", [CLEARED, "literal changed output"], ids=["placeholder", "ordinary-text"])
def test_1179_steady_skip_requires_literal_content(build: Build, content: str) -> None:
    engine = build()
    host = fixture()[-3:]
    engine.ingest(deepcopy(host))
    before = stored(engine)
    host[0]["timestamp"] = 20
    host[-1]["content"] = content
    engine.ingest(host)
    after = stored(engine)
    assert after[:len(before)] == before
    added = after[len(before):]
    assert sum(row["role"] == "user" and row["content"] == "repeat" for row in added) == 1
    assert sum(row["role"] == "tool" and row["content"] == content for row in added) == 1
    # The unchanged call may retain its existing occurrence; the changed result may not.
    if content == CLEARED:
        assert any(hashlib.sha256(row["content"].encode()).hexdigest() == CLEARED_SHA256 for row in added)
