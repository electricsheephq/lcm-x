"""Offline public contracts for previously unobserved auxiliary-client adapters."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

extraction = importlib.import_module("hermes_lcm.assertion_extraction")
selective = importlib.import_module("hermes_lcm.selective_compiler")
assertions = importlib.import_module("hermes_lcm.assertion_store")
messages = importlib.import_module("hermes_lcm.store")
compiler = importlib.import_module("hermes_lcm.evidence_compiler")
harness = importlib.import_module("tests.harness")


def install_auxiliary(monkeypatch: pytest.MonkeyPatch, content: Any, usage: Any) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    def call_llm(**kwargs: Any) -> SimpleNamespace:
        calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
            usage=usage, provider="fixture", model="fixture-model",
        )

    module = ModuleType("agent.auxiliary_client")
    module.call_llm = call_llm  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "agent.auxiliary_client", module)
    return calls


@pytest.mark.parametrize("usage, expected", [
    ({"prompt_tokens": "17", "completion_tokens": 9}, (17, 9)),
    (SimpleNamespace(input_tokens=4, output_tokens=3), (4, 3)),
    ({"input_tokens": "bad", "output_tokens": None}, (0, 0)),
    (None, (0, 0)),
])
def test_assertion_extractor_uses_auxiliary_reply_and_usage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, usage: Any, expected: tuple[int, int],
) -> None:
    store = messages.MessageStore(tmp_path / "assertions.db")
    derived = assertions.AssertionStore(store.db_path)
    try:
        store_id = store.append("s", {"role": "user", "content": "Tea is available."})
        snapshot = derived.snapshot_source(store_id)
        payload = json.dumps({
            "schema_version": 1, "source_store_id": store_id,
            "source_content_sha256": snapshot.content_sha256,
            "assertions": [], "relations": [],
        })
        calls = install_auxiliary(monkeypatch, "<think>private</think>" + payload, usage)
        extractor = extraction.ModelAssertionExtractor(derived, timeout_seconds=2)
        result = extractor(snapshot)
        assert result.assertions == result.relations == ()
        assert (extractor.total_input_tokens, extractor.total_output_tokens) == expected
        assert len(calls) == 1
        assert calls[0]["task"] == "assertion_extraction"
        assert calls[0]["timeout"] == 2
        assert "EXACT_SOURCE:" in calls[0]["messages"][0]["content"]
        assert store.get(store_id)["content"] == "Tea is available."
    finally:
        derived.close()
        store.close()


@pytest.mark.parametrize("reply", [None, 42, "<think>only reasoning</think>"])
def test_assertion_extractor_rejects_invalid_auxiliary_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reply: Any,
) -> None:
    store = messages.MessageStore(tmp_path / "assertions.db")
    derived = assertions.AssertionStore(store.db_path)
    try:
        sid = store.append("s", {"role": "user", "content": "Keep this source."})
        install_auxiliary(monkeypatch, reply, None)
        extractor = extraction.ModelAssertionExtractor(derived)
        with pytest.raises(ValueError):
            extractor(derived.snapshot_source(sid))
        assert not derived.has_current_receipt(derived.snapshot_source(sid))
        assert store.get(sid)["content"] == "Keep this source."
    finally:
        derived.close()
        store.close()


@pytest.mark.parametrize("usage, expected", [
    ({"prompt_tokens": "7", "completion_tokens": 3}, (7, 3)),
    (SimpleNamespace(input_tokens=-2, output_tokens=6), (0, 6)),
    ({"input_tokens": float("inf"), "output_tokens": "bad"}, (0, 0)),
    (None, (0, 0)),
])
def test_selective_auxiliary_decodes_fence_and_reports_bounded_usage(
    monkeypatch: pytest.MonkeyPatch, usage: Any, expected: tuple[int, int],
) -> None:
    calls = install_auxiliary(monkeypatch, '<think>private</think>```json\n{"selections": []}\n```', usage)
    monkeypatch.setattr(selective.time, "perf_counter", lambda: 5.0)
    proposal, provenance = selective.call_selective_auxiliary_selector(
        {"status": "selector_required", "prompt": "Choose exact evidence."}, timeout_seconds=99,
    )
    assert proposal == {"selections": []}
    assert (provenance["input_tokens"], provenance["output_tokens"]) == expected
    assert provenance["provider"] == "fixture"
    assert provenance["model"] == "fixture-model"
    assert provenance["calls"] == 1
    assert calls[0]["timeout"] == provenance["timeout_seconds"] == 8.0
    assert calls[0]["messages"] == [{"role": "user", "content": "Choose exact evidence."}]


@pytest.mark.parametrize("prepared", [{}, {"status": "selector_required"}])
def test_selective_selector_refuses_unprepared_request(prepared: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="not required"):
        selective.call_selective_auxiliary_selector(prepared)


def test_selective_selector_rejects_array_reply(monkeypatch: pytest.MonkeyPatch) -> None:
    install_auxiliary(monkeypatch, "[]", None)
    with pytest.raises(ValueError, match="non-object"):
        selective.call_selective_auxiliary_selector({"status": "selector_required", "prompt": "p"})


@pytest.mark.parametrize("number, expected", [("two", 2), ("10", 10)])
def test_explicit_question_cardinality_survives_request_derivation(number: str, expected: int) -> None:
    request = compiler.derive_evidence_request(f"What is the total cost of the {number} invoices?")
    assert request["expected_cardinality"] == expected


def test_rebuild_plan_reports_remaining_sources(tmp_path: Path) -> None:
    store = messages.MessageStore(tmp_path / "assertions.db")
    derived = assertions.AssertionStore(store.db_path)
    try:
        store.append_batch("s", [{"role": "user", "content": "first"}, {"role": "assistant", "content": "second"}])
        plan = derived.plan_rebuild(limit=1)
        assert plan.pending_count == 2
        assert len(plan.selected_sources) == plan.remaining_count == 1
        derived.publish_source(plan.selected_sources[0], [])
        next_plan = derived.plan_rebuild(limit=1)
        assert next_plan.pending_count == 1
        assert next_plan.remaining_count == 0
    finally:
        derived.close()
        store.close()
