"""Pin recent-family response bytes and failure logging before extraction."""
from __future__ import annotations

import inspect
import json
import logging
from types import SimpleNamespace
from typing import Any, cast

import pytest

from hermes_lcm import recent as recent_module
from hermes_lcm import evidence_tools, inspect_tools
from hermes_lcm import tools
from hermes_lcm.engine import LCMEngine
from hermes_lcm.rollup_periods import parse_recent_period


@pytest.mark.parametrize("fallback", [False, True])
def test_recent_facade_response_bytes(
    monkeypatch: pytest.MonkeyPatch, fallback: bool,
) -> None:
    engine = SimpleNamespace(current_session_id="current")
    rows: list[dict[str, Any]] = [{
        "rollup_id": 7, "period_kind": "day", "period_start": "2026-07-15",
        "status": "ready", "token_count": 2, "summary": "remember café",
    }]
    leaf: dict[str, Any] = {
        "kind": "leaf_summary", "node_id": 9, "session_id": "current",
        "token_count": 2, "earliest_at": 1.0, "latest_at": 2.0,
        "content": "remember café", "content_truncated": False,
    }
    calls: list[str] = []

    def ready(*args: Any) -> tuple[list[dict[str, Any]], str | None]:
        calls.append("ready")
        return ([], "rollups_unavailable") if fallback else (rows, None)

    def leaves(*args: Any) -> list[dict[str, Any]]:
        calls.append("leaves")
        return [leaf]

    monkeypatch.setattr(recent_module, "_recent_ready_rollups", ready)
    monkeypatch.setattr(recent_module, "_recent_leaf_sections", leaves)
    raw = tools.lcm_recent({"period": "date:2026-07-15", "limit": 500}, engine=engine)
    section = leaf if fallback else {
        "kind": "rollup", "rollup_id": 7, "period_kind": "day",
        "period_start": "2026-07-15", "status": "ready", "token_count": 2,
        "content": "Tokens: 2\nremember café", "content_truncated": False,
    }
    expected: dict[str, Any] = {
        "period": "date:2026-07-15", "scope": "conversation",
        "window": {"start": "2026-07-15T00:00:00Z", "end": "2026-07-16T00:00:00Z"},
        "limit": 200, "char_limit": 20_000,
        "mode": "leaf_summary_fallback" if fallback else "rollup",
        "provenance": {
            "fallback": fallback,
            "rollups": [] if fallback else [{"rollup_id": 7, "status": "ready"}],
        },
        "rollups_covered": 0 if fallback else 1,
    }
    if fallback:
        expected["fallback_reason"] = "rollups_unavailable"
    expected.update(limit_clamped_from=500, sections=[section], total_sections=1,
                    returned_sections=1, truncated=False)
    assert raw == json.dumps(expected, ensure_ascii=False)
    assert calls == (["ready", "leaves"] if fallback else ["ready"])


def test_recent_rollup_failure_keeps_logger_and_fallback(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
) -> None:
    engine = cast(LCMEngine, SimpleNamespace(
        current_session_id="current", _dag=SimpleNamespace(db_path="unused.db"),
        _config=SimpleNamespace(temporal_rollups_enabled=True),
    ))

    def unavailable(*args: Any) -> None:
        raise RuntimeError("characterized read failure")

    monkeypatch.setattr(recent_module, "RollupStore", unavailable)
    with caplog.at_level(logging.DEBUG, logger="hermes_lcm.tools"):
        assert recent_module._recent_ready_rollups(
            engine, parse_recent_period("date:2026-07-15"), "current",
        ) == ([], "rollups_unavailable")
    assert [(r.name, r.levelname, r.message) for r in caplog.records] == [(
        "hermes_lcm.tools", "DEBUG", "LCM recent rollup read failed; using leaf summaries",
    )]
    assert caplog.records[0].exc_info is not None


def test_recent_scope_preserves_current_and_finalized_order() -> None:
    engine = cast(LCMEngine, SimpleNamespace(
        current_session_id="current", current_conversation_id="conversation",
        _lifecycle=SimpleNamespace(get_by_conversation=lambda _: SimpleNamespace(
            current_session_id="current", last_finalized_session_id="finalized",
        )),
    ))
    assert recent_module._recent_conversation_scope_session_ids(engine) == [
        "current", "finalized",
    ]


def test_recent_missing_engine_error_bytes() -> None:
    assert tools.lcm_recent({}) == '{"error": "LCM engine not initialized"}'


def test_tools_facade_does_not_reexport_recent_private_helpers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Public tool facades and other modules' load-bearing exports are allowed.
    for family in (recent_module, evidence_tools, inspect_tools):
        private_helpers = {
            name for name, value in vars(family).items()
            if name.startswith("_") and inspect.isfunction(value)
            and value.__module__ == family.__name__
        }
        assert private_helpers
        assert private_helpers.isdisjoint(vars(tools))
        assert not {
            name for name, value in vars(tools).items()
            if name.startswith("_") and inspect.isfunction(value)
            and value.__module__ == family.__name__
        }
    assert callable(tools.lcm_recent)

    # Keep this existing guard's ID while characterizing the next facade family.
    for handler in (
        tools.lcm_query_state, tools.lcm_compute, tools.lcm_evidence_pack,
        tools.lcm_compile_evidence, tools.lcm_retrieve,
    ):
        assert handler({}) == '{"error": "LCM engine not initialized"}'
        assert handler.__module__ == "hermes_lcm.tools"

    engine = SimpleNamespace()
    assert tools.lcm_query_state({}, engine=engine) == json.dumps({
        "status": "disabled",
        "error": "V4 assertions are not enabled for this profile",
    })
    assert tools.lcm_compile_evidence({"mode": "invalid"}, engine=engine) == (
        '{"error": "mode must be one of: proposal, auto"}'
    )
    assert tools.lcm_retrieve({}, engine=engine) == json.dumps({
        "status": "disabled",
        "reason": "adaptive retrieval is disabled",
        "enable_with": "LCM_ADAPTIVE_RETRIEVAL_ENABLED=true",
        "provenance": {"controller": {
            "transport": "deterministic_local", "provider": "none", "model": "none",
        }},
    })

    from hermes_lcm import evidence_compiler, evidence_pack

    calls: list[dict[str, Any]] = []

    def recall(args: dict[str, Any], **kwargs: Any) -> str:
        assert kwargs["engine"] is engine
        calls.append(args)
        return "exact café evidence"

    def pack(args: dict[str, Any], **kwargs: Any) -> str:
        assert kwargs["engine"] is engine
        return str(kwargs["retrieve"]({"query": args["question"]}))

    def compile_result(question: str, **kwargs: Any) -> dict[str, Any]:
        assert kwargs["engine"] is engine
        return {"evidence": kwargs["retrieve"]({"query": question})}

    monkeypatch.setattr(tools, "lcm_recall", recall)
    monkeypatch.setattr(evidence_pack, "build_evidence_pack", pack)
    monkeypatch.setattr(evidence_compiler, "compile_evidence", compile_result)
    monkeypatch.setattr(evidence_compiler, "compile_preanswer_evidence", compile_result)
    assert tools.lcm_evidence_pack({"question": "bounded"}, engine=engine) == (
        "exact café evidence"
    )
    for mode in ("proposal", "auto"):
        assert tools.lcm_compile_evidence(
            {"question": "bounded", "mode": mode}, engine=engine,
        ) == '{"evidence": "exact café evidence"}'
    assert calls == [{"query": "bounded"}] * 3
