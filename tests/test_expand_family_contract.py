"""Pin describe/expand bytes, read order and late-bound facade dependencies."""
from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from hermes_lcm import tools
from hermes_lcm.dag import SummaryDAG, SummaryNode
from hermes_lcm.store import MessageStore
from tests.harness import pin_char_counter


@pytest.fixture
def inventory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Any]:
    pin_char_counter(monkeypatch)  # type: ignore[no-untyped-call]
    store: Any = MessageStore(tmp_path / "store.db")
    dag: Any = SummaryDAG(tmp_path / "store.db")
    for i in (1, 2, 3):
        store.append("session" if i < 3 else "archive", dict(
            role="user", content="café evidence " * i, timestamp=100), source="cli")
    store._conn.execute("UPDATE messages SET timestamp = 100")
    store._conn.commit()
    for i, kind, sources in ((1, "messages", [1, 2]), (2, "messages", [3]),
                             (3, "nodes", [1, 2]), (4, "unknown", [])):
        dag.add_node(SummaryNode(session_id="session", depth=int(i == 3),
                                summary=f"café summary {i}", token_count=4,
                                source_token_count=32, source_ids=sources, source_type=kind,
                                created_at=100, earliest_at=100, latest_at=100,
                                expand_hint=f"lcm_expand(node_id={i})"),
                     escalation_level=1 if i == 3 else None, model="fixture")
    payload = dict(content="café payload " * 50, session_id="session", created_at=100,
                   content_chars=650, content_bytes=700, kind="tool_result")
    monkeypatch.setattr(tools, "load_externalized_payload",
                        lambda ref, **kw: payload if ref == "payload.txt" else None)
    try:
        yield SimpleNamespace(current_session_id="session", _store=store, _dag=dag,
                              _config=SimpleNamespace(), _hermes_home=str(tmp_path))
    finally:
        dag.close()
        store.close()


def response_bytes(engine: Any, monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    responses: dict[str, str] = {}
    for name in ("lcm_describe", "lcm_expand"):
        handler = getattr(tools, name)
        responses[name + "/missing_engine"] = handler({})
        args_list: list[dict[str, Any]] = [
            {}, {"node_id": 3}, {"node_id": 1}, {"store_id": 1},
            {"externalized_ref": "payload.txt"}, {"externalized_ref": "missing.txt"},
            {"externalized_ref": "payload.txt", "session_id": "archive"},
            {"node_id": 99}, {"node_id": 99, "session_id": "archive"},
            {"node_id": 1, "session_id": "archive"}, {"node_id": 3, "session_id": " session "},
            {"session_id": "session"}, {"session_id": ""}, {"session_id": 1},
        ]
        if name == "lcm_expand":
            store_ids: list[Any] = ["bad", [], float("inf"), 99, 3]
            args_list += [
                {"node_id": 1, "store_id": 1}, {"store_id": 1, "session_id": "archive"},
                {"node_id": 1, "session_id": ""}, {"node_id": 1, "session_id": 1},
                {"node_id": 1, "include_exact_ref": "yes"},
                {"node_id": 1, "include_exact_ref": True}, {"node_id": 4},
                *({"store_id": value} for value in store_ids),
                {"store_id": 1, "include_exact_ref": True, "max_tokens": 3},
                {"store_id": 1, "include_exact_ref": True, "content_offset": 999},
                {"node_id": 3, "source_limit": 1}, {"node_id": 3, "source_offset": 1},
                {"node_id": 3, "max_tokens": 2}, {"node_id": 1, "max_tokens": 3},
                {"externalized_ref": "payload.txt", "max_tokens": 3},
                {"externalized_ref": "payload.txt", "content_offset": 3},
            ]
        for i, args in enumerate(args_list):
            responses[f"{name}/{i}"] = handler(args, engine=engine)
    first = json.loads(tools.lcm_expand({"node_id": 1, "max_tokens": 3}, engine=engine))["pagination"]
    responses["continuation"] = tools.lcm_expand(dict(
        node_id=1, max_tokens=3, source_offset=first["next_source_offset"],
        content_offset=first["next_content_offset"]), engine=engine)
    row = engine._store.get(1)
    row["content"] = "[Externalized tool output: payload.txt]"
    row["tool_calls"] = {"opaque": {1}}
    with monkeypatch.context() as patch:
        patch.setattr(engine._store, "get", lambda _: row)
        for session in ("session", "archive"):
            row["session_id"] = session
            responses["reference/" + session] = tools.lcm_expand({"store_id": 1}, engine=engine)
    with monkeypatch.context() as patch:
        patch.setattr(engine._store, "get", lambda _: row)
        row["content"] = "[Externalized tool output: chars=650; ref=payload.txt]"
        row["tool_calls"] = {"ref": "[Externalized LCM ingest payload: chars=1; ref=missing.txt]"}
        for session in ("session", "archive"):
            row["session_id"] = session
            responses["metadata/" + session] = tools.lcm_expand({"store_id": 1}, engine=engine)
    return responses


def test_response_bytes(inventory: Any, monkeypatch: pytest.MonkeyPatch,
                        caplog: pytest.LogCaptureFixture) -> None:
    expected = json.loads((Path(__file__).parent / "fixtures" / "expand_family_contract.json").read_text())
    with caplog.at_level(logging.DEBUG, logger="hermes_lcm.tools"):
        assert response_bytes(inventory, monkeypatch) == expected
    assert not [r for r in caplog.records if r.name == "hermes_lcm.tools"]


def test_read_order(inventory: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str] = []
    def wrap(name: str, original: Any) -> Any:
        def marker(*args: Any, **kwargs: Any) -> Any:
            seen.append(name)
            return original(*args, **kwargs)
        return marker
    for owner, names in ((inventory._dag, ("get_node", "describe_subtree", "get_node_provenance",
                                          "get_session_depth_stats", "get_session_depth_samples")),
                         (inventory._store, ("get", "get_batch", "get_session_count"))):
        for name in names:
            monkeypatch.setattr(owner, name, wrap(name, getattr(owner, name)))
    for handler, args, expected in (
        (tools.lcm_describe, {}, ["get_session_depth_stats", "get_session_depth_samples", "get_session_count"]),
        (tools.lcm_describe, {"node_id": 1}, ["get_node", "describe_subtree", "get_node", "get_node_provenance"]),
        (tools.lcm_expand, {"node_id": 1}, ["get_node", "get_batch"]),
        (tools.lcm_expand, {"node_id": 3}, ["get_node", "get_node", "get_node"]),
        (tools.lcm_expand, {"store_id": 1}, ["get"]),
    ):
        seen.clear()
        handler(args, engine=inventory)
        assert seen == expected


@pytest.mark.parametrize("handler,args,owner,method", [
    ("lcm_describe", {"node_id": 1}, "_dag", "describe_subtree"),
    ("lcm_describe", {}, "_dag", "get_session_depth_stats"),
    ("lcm_expand", {"store_id": 1}, "_store", "get"),
    ("lcm_expand", {"node_id": 1}, "_store", "get_batch"),
])
def test_read_failure_propagates_without_logging(inventory: Any, monkeypatch: pytest.MonkeyPatch,
                                                caplog: pytest.LogCaptureFixture, handler: str,
                                                args: dict[str, Any], owner: str, method: str) -> None:
    def fail(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("characterized read failure")
    monkeypatch.setattr(getattr(inventory, owner), method, fail)
    with caplog.at_level(logging.DEBUG, logger="hermes_lcm.tools"):
        with pytest.raises(RuntimeError, match="characterized read failure"):
            getattr(tools, handler)(args, engine=inventory)
    assert not [r for r in caplog.records if r.name == "hermes_lcm.tools"]


def test_shared_patch_points(inventory: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    shared = ("_require_engine", "_get_externalized_payload", "_get_session_node",
              "_session_expand_hint", "_parse_positive_int", "_parse_non_negative_int",
              "_slice_content_for_response", "extract_ingest_externalized_refs",
              "extract_externalized_ref", "_expand_message_sources", "_expand_child_nodes")
    seen: set[str] = set()
    def wrap(name: str, original: Any) -> Any:
        def marker(*args: Any, **kwargs: Any) -> Any:
            seen.add(name)
            return original(*args, **kwargs)
        return marker
    for name in shared:
        monkeypatch.setattr(tools, name, wrap(name, getattr(tools, name)))
    response_bytes(inventory, monkeypatch)
    assert seen == set(shared)
    # Exercise the JSON module binding too, without changing the shared stdlib module.
    monkeypatch.setattr(tools, "json", SimpleNamespace(dumps=lambda *a, **k: "json marker"))
    assert tools.lcm_describe({}) == "json marker"
    assert tools.lcm_expand({}) == "json marker"
