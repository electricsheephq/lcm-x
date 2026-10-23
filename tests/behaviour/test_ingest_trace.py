"""Ingest's frozen observable protocol and independent observer inventory."""
from __future__ import annotations

import json
import hashlib
import re
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import pytest

from tests.behaviour import ingest_trace_harness as harness
from tests.behaviour.ingest_trace_harness import (
    GOLDEN, OBSERVED_ATTRIBUTES, SCENARIOS, TRACE_GUIDANCE, IngestCanonical,
    engine_attributes, engine_module, first_difference, observe,
    install_clock, reopen, search_observation, snapshot,
)
from tests.harness import engine_context, pin_char_counter


@pytest.mark.parametrize("case", SCENARIOS, ids=lambda case: case.name)
def test_ingest_golden(case: Any, tmp_path: Path) -> None:
    expected: dict[str, list[str]] = json.loads(GOLDEN.read_text())
    payloads: list[str] = []
    actual = observe(case, tmp_path, payloads=payloads)
    difference = first_difference(expected[case.name], actual)
    cache_diagnostics = []
    if difference is not None:
        (tmp_path / "actual-payload.json").write_text(json.dumps(payloads, ensure_ascii=False))
        for payload in payloads:
            stage, marker, data = payload.partition("attr _no_progress_public_rows ")
            if marker and isinstance(cache := json.loads(data), dict):
                entries = {}
                for key, value in cache.items():
                    label = key if re.fullmatch(r"(?:retained-message|message|cached-occurrence)-\d+", key) else "<unlabelled>"
                    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"))
                    entries[label] = dict(sha12=hashlib.sha256(encoded.encode()).hexdigest()[:12],
                                         fields=len(value) if isinstance(value, dict) else 0, chars=len(encoded))
                cache_diagnostics.append(dict(stage=stage, entries=entries))
    assert difference is None, f"{case.name}: {difference}\n{TRACE_GUIDANCE}\ncache-metadata: {json.dumps(cache_diagnostics, sort_keys=True)}"


def test_observer_completeness(tmp_path: Path) -> None:
    discovered, methods = engine_attributes(Path(engine_module.__file__).parent)
    assert "_ingest_messages" in methods
    assert discovered <= OBSERVED_ATTRIBUTES, f"Unobserved engine state: {sorted(discovered - OBSERVED_ATTRIBUTES)}"
    assert "_ingest_cursor" in discovered
    assert "_identity_anchor_commit" in methods
    # The closure follows a future extracted phase and that phase's helper.
    (tmp_path / "engine.py").write_text(
        "class LCMEngine:\n"
        " def _ingest_messages(self): return self._moved_phase()\n"
        " def _moved_phase(self): return self._helper()\n"
        " def _helper(self): return self._trace_inventory_probe\n")
    extra, reachable = engine_attributes(tmp_path)
    assert extra == {"_trace_inventory_probe"}
    assert reachable == {"_ingest_messages", "_moved_phase", "_helper"}
    assert not extra <= OBSERVED_ATTRIBUTES


def test_fixture_scenario_inventory_and_size() -> None:
    data: dict[str, list[str]] = json.loads(GOLDEN.read_text())
    names = [case.name for case in SCENARIOS]
    assert len(names) == 202
    assert len(names) == len(set(names))
    assert set(data) == set(names), TRACE_GUIDANCE
    # Initial, next and cold ingests now each retain their full observer stream.
    # Parts A/B both add transient ingest observations.
    assert GOLDEN.stat().st_size <= 3_400_000
    assert all(len(json.dumps(trace).encode()) <= 45_000 for trace in data.values())


def test_first_divergence() -> None:
    assert first_difference(["BEGIN", "row", "COMMIT"], ["BEGIN", "row", "ROLLBACK"]) == (
        "first divergence event 2: expected 'COMMIT', got 'ROLLBACK'")
    assert first_difference(["row"], []) == "first divergence event 0: expected 'row', got '<end>'"
    assert first_difference([], []) is None


@pytest.mark.parametrize("name", ["cache-new-uid", "anchor-stamp-new-uid", "proof-output-new-uid",
                                  "proof-native-new-uid", "proof-consumed-new-uid"])
def test_new_uid_seed_matches_first_immutable_fact(name: str, tmp_path: Path) -> None:
    payloads: list[str] = []
    observe(next(case for case in SCENARIOS if case.name == name), tmp_path, payloads=payloads)
    states = {label: json.loads(next(event[len(label) + 1:] for event in payloads
                                    if event.startswith(label + " ")))
              for label in ("state-before", "state-after", "next-state", "cold-state")}
    metadata = dict(states["state-before"]["tables"]["metadata"])
    if name == "proof-native-new-uid":
        assert not any(key.startswith(("host_occurrence_row:", "host_occurrence_accepted_view:"))
                       for key in metadata)
    else:
        assert json.loads(metadata["host_occurrence_row:S:1"])["message_uid"] == "old-occurrence"
    before = states["state-before"]["tables"]["messages"]
    after = states["state-after"]["tables"]["messages"]
    assert after[:len(before)] == before
    assert len(after) > len(before)
    incoming = json.loads(next(event.removeprefix("input-before ") for event in payloads
                               if event.startswith("input-before ")))
    assert incoming[0]["message_uid"] == "fresh-occurrence"
    assert states["next-state"]["tables"]["messages"] == after
    assert states["cold-state"]["tables"]["messages"] == after


@pytest.mark.parametrize("name", ["proof-native-remap", "proof-native-remap-new-stamp"])
def test_native_manual_proof_is_legacy_reader_without_extra_note(name: str, tmp_path: Path) -> None:
    payloads: list[str] = []
    observe(next(case for case in SCENARIOS if case.name == name), tmp_path, payloads=payloads)
    before = json.loads(next(event.removeprefix("state-before ") for event in payloads
                             if event.startswith("state-before ")))
    assert not any(row[0].startswith(("host_occurrence_row:", "host_occurrence_accepted_view:"))
                   for row in before["tables"]["metadata"])
    for label in ("state-after", "next-state", "cold-state"):
        state = json.loads(next(event[len(label) + 1:] for event in payloads
                                if event.startswith(label + " ")))
        assert not any(row[5] == harness.NOTE["content"] for row in state["tables"]["messages"])
        assert state["tables"]["messages"][:4] == before["tables"]["messages"]


@pytest.mark.parametrize("name", ["uid-bindings-new-occurrence", "uid-bindings-canonical-same-row"])
def test_uid_stamp_partition_has_actual_same_row_witness(name: str, tmp_path: Path) -> None:
    payloads: list[str] = []
    observe(next(case for case in SCENARIOS if case.name == name), tmp_path, payloads=payloads)
    def event(label: str) -> Any:
        return json.loads(next(row[len(label) + 1:] for row in payloads if row.startswith(label + " ")))
    before, after = event("state-before"), event("state-after")
    metadata = dict(before["tables"]["metadata"])
    bindings = before["tables"]["host_uid_bindings"]
    assert len(bindings) == 4 and all(row[3] == "canonical" for row in bindings)
    assert event("input-before")[0]["timestamp"] == event("input-after")[0]["timestamp"] == 999.
    rows = after["tables"]["messages"]
    assert rows[:4] == before["tables"]["messages"]
    if name == "uid-bindings-new-occurrence":
        assert not any(key.startswith(("host_occurrence_row:", "host_occurrence_accepted_view:"))
                       for key in metadata)
        assert len(rows) == 5 and event("return-0")[0]["timestamp"] == 999.
    else:
        fact = json.loads(metadata["host_occurrence_row:S:1"])
        assert fact["message_uid"] == event("input-before")[0]["message_uid"] == "host-uid-0"
        assert any(row[0] == 1 and row[1] == "host-uid-0" and row[2] for row in bindings)
        assert rows[0][5] == event("input-before")[0]["content"]
        assert len(rows) == 4 and event("return-0")[0]["timestamp"] == 100.
    assert event("cold-return")[0]["timestamp"] == 999.
    for label in ("next-state", "cold-state"):
        assert event(label)["tables"]["messages"] == rows
        assert event(label)["tables"]["host_uid_bindings"] == after["tables"]["host_uid_bindings"]


def test_exact_evidence_digest_mapping_keeps_content_distinctions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    canonical = IngestCanonical(tmp_path, [])
    harness.observe_occurrence_digests(monkeypatch, canonical)
    identities = [("user", str(tmp_path / "payload.txt") + suffix, "", "", "")
                  for suffix in ("", " ", " changed")]
    raw = [engine_module._commit_proof_identity_digest(identity) for identity in identities]
    clean = [canonical.value(digest, "input_sha256") for digest in raw]
    assert len(set(raw)) == len(set(clean)) == 3
    assert clean == [hashlib.sha256(json.dumps(list(canonical.value(identity, "content")),
                    ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
                    for identity in identities]
    assert canonical.value({"content": raw[0], "message_uid": raw[1]}) == {
        "content": raw[0], "message_uid": raw[1]}
    assert canonical.value("f" * 64, "input_sha256") == "f" * 64
    assert all(row["raw"] != row["canonical"] for row in canonical.digest_sources.values())
    with pytest.MonkeyPatch.context() as collision:
        collision.setattr(engine_module, "_commit_proof_identity_digest", lambda identity: "0" * 64)
        harness.observe_occurrence_digests(collision, IngestCanonical(tmp_path, []))
        engine_module._commit_proof_identity_digest(identities[0])
        with pytest.raises(AssertionError, match="inconsistent evidence digest"):
            engine_module._commit_proof_identity_digest(identities[1])


def test_occurrence_uids_and_structured_payload_bytes_stay_distinct(tmp_path: Path) -> None:
    canonical = IngestCanonical(tmp_path, [])
    first = {"role": "user", "content": {"mtime_ns": 100, "message_uid": "nested"},
             "message_uid": "12345678123412341234123456789abc", "timestamp": 10.}
    second = {**first, "message_uid": "fedcba9876543210fedcba9876543210"}
    assert canonical.value(first) != canonical.value(second)
    assert canonical.value(first)["content"] == first["content"]
    assert canonical.value({"content": "mtime_ns=123"}) == {"content": "mtime_ns=123"}
    assert canonical.value(first) != canonical.value({**first, "timestamp": 11.})
    replay = canonical.value(("S", "C", "sandbox-digest-a"), "_no_progress_replay")
    assert replay == canonical.value(("S", "C", "sandbox-digest-a"), "_no_progress_replay")
    assert replay != canonical.value(("S", "C", "sandbox-digest-b"), "_no_progress_replay")
    assert canonical.value({id(first): first}, "_no_progress_public_rows") == {
        "cached-occurrence-0": canonical.value(first)}


def test_search_observation_detects_unindexed_ingest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pin: Any = pin_char_counter
    context: Any = engine_context
    pin(monkeypatch)
    install_clock(monkeypatch)
    with context(tmp_path, window=200000, conversation_id="trace-conversation", embeddings_enabled=False) as engine:
        first = [{"role": "user", "content": "indexed trace observation"}]
        engine._ingest_messages(first)
        assert search_observation(engine, first) == [[first[0]["content"], [[1, "S", first[0]["content"]]]]]
        # A real SQLite trigger failure leaves durable bytes but no FTS entry.
        engine._store._conn.execute("DROP TRIGGER msg_fts_insert")
        missing = {"role": "user", "content": "unindexed distinct needle"}
        engine._ingest_messages([*first, missing])
        observations = dict(search_observation(engine, [missing]))
        assert engine._store.get_session_count("S") == 2
        assert observations[missing["content"]] == []


def test_reopen_discards_instance_caches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pin: Any = pin_char_counter
    context: Any = engine_context
    pin(monkeypatch)
    install_clock(monkeypatch)
    with context(tmp_path, window=200000, conversation_id="trace-conversation", embeddings_enabled=False) as old, ExitStack() as engines:
        old._host_uid_lineage_cache = {"stale": True}
        old._host_uid_counters = {"stale": 7}
        old._engine_uids_pending = {"stale": True}
        fresh = reopen(old, tmp_path, engines)
        assert fresh is not old
        assert fresh._config is old._config
        for name in ("_host_uid_lineage_cache", "_host_uid_counters", "_engine_uids_pending"):
            assert "stale" not in getattr(fresh, name, {})


def test_snapshot_keeps_labelled_messages_alive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pin: Any = pin_char_counter
    context: Any = engine_context
    pin(monkeypatch)
    install_clock(monkeypatch)
    with context(tmp_path, window=200000, embeddings_enabled=False) as engine:
        message = {"role": "assistant", "content": "retained identity"}
        engine._last_active_replay_messages = [message]
        canonical = IngestCanonical(tmp_path, [])
        snapshot(engine, canonical)
        engine._last_active_replay_messages = []
        assert canonical.retained_messages[id(message)] is message


def test_copy_return_observer_retains_replaced_owners(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    owners = {}
    harness.retain_replay_copy_owners(monkeypatch, owners)
    with engine_context(tmp_path, window=200000, embeddings_enabled=False) as engine:
        def copy_occurrence(content):
            rows = engine._copy_active_replay_messages_preserving_generated_ids([
                {"role": "assistant", "content": content}])
            return id(rows[0])

        first = copy_occurrence("first synthetic occurrence")
        second = copy_occurrence("second synthetic occurrence")
        assert first != second
        assert owners[first]["content"] == "first synthetic occurrence"
        assert owners[second]["content"] == "second synthetic occurrence"
        assert set(engine._no_progress_public_rows) == {second}


@pytest.mark.parametrize("name", [case.name for case in SCENARIOS if case.shape == "uid"])
def test_uid_scenarios_bind_durable_rows(name: str, tmp_path: Path) -> None:
    payloads: list[str] = []
    observe(next(case for case in SCENARIOS if case.name == name), tmp_path, payloads=payloads)
    state = json.loads(next(event.removeprefix("state-after ") for event in payloads
                            if event.startswith("state-after ")))
    bindings = state["tables"]["host_uid_bindings"]
    assert len(bindings) == (1 if name == "uid-bindings-new" else 4)
    assert all(row[3] == "canonical" for row in bindings)
    assert state["_host_uid_counters"]["unbound.agree_new"] == len(bindings)
    assert not any(key.startswith("skipped.") for key in state["_host_uid_counters"])


@pytest.mark.parametrize("damage", ["truncate", "delete"])
def test_payload_observation_detects_damaged_externalization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, damage: str,
) -> None:
    pin: Any = pin_char_counter
    context: Any = engine_context
    pin(monkeypatch)
    install_clock(monkeypatch)
    with context(tmp_path, window=200000, conversation_id="trace-conversation", embeddings_enabled=False,
                 large_output_externalization_enabled=True, large_output_externalization_threshold_chars=100,
                 large_output_active_replay_stubbing_enabled=True,
                 large_output_active_replay_stub_threshold_tokens=10,
                 large_output_externalization_path=str(tmp_path / "home" / "payloads")) as engine:
        incoming = [{"role": "tool", "tool_call_id": "call-1", "content": "durable output words " * 100}]
        engine._ingest_messages(incoming)
        canonical = IngestCanonical(tmp_path, incoming)
        before = canonical.value(harness.payload_observation(engine))
        assert len(before) == 1
        entry = before[0]
        path = tmp_path / "home" / "payloads" / entry["ref"]
        raw = path.read_bytes()
        assert entry["exists"] is True
        assert entry["bytes"] == len(raw)
        assert entry["sha256"] == hashlib.sha256(raw).hexdigest()
        assert entry["read"]["content"] == incoming[0]["content"]
        assert entry["read"]["content_truncated"] is False
        if damage == "truncate":
            path.write_bytes(b"{")
        else:
            path.unlink()
        after = canonical.value(harness.payload_observation(engine))
        assert after != before
        assert after[0]["ref"] == entry["ref"]
        assert after[0]["exists"] is (damage == "truncate")
        assert after[0]["bytes"] == (1 if damage == "truncate" else None)
        assert "error" in after[0]["read"]
        with ExitStack() as engines:
            fresh = reopen(engine, tmp_path, engines)
            assert canonical.value(harness.payload_observation(fresh)) == after


@pytest.mark.parametrize("name", [case.name for case in SCENARIOS if case.shape == "tool-stub"])
def test_externalized_manifest_recorded_after_ingest_and_cold_reopen(name: str, tmp_path: Path) -> None:
    payloads: list[str] = []
    observe(next(case for case in SCENARIOS if case.name == name), tmp_path, payloads=payloads)
    for label in ("payloads-after", "payloads-cold"):
        manifest = json.loads(next(event.removeprefix(label + " ") for event in payloads
                                   if event.startswith(label + " ")))
        assert len(manifest) == (2 if name in {"tool-stub-pair", "tool-stub-host-held"} else 1)
        assert all(entry["exists"] and entry["bytes"] > 0 and len(entry["sha256"]) == 64
                   and entry["read"]["content"] for entry in manifest)


def test_reopen_preserves_bound_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pin: Any = pin_char_counter
    context: Any = engine_context
    pin(monkeypatch)
    install_clock(monkeypatch)
    with context(tmp_path, window=131072, platform="cli", conversation_id="bound-conversation",
                 embeddings_enabled=False) as old, ExitStack() as engines:
        old._ingest_messages([{"role": "user", "content": "before reopen"}])
        fresh = reopen(old, tmp_path, engines)
        fresh._ingest_messages([{"role": "user", "content": "after reopen"}])
        rows = fresh._store.get_session_messages("S")
        assert len(rows) == 2
        assert rows[0]["source"] == rows[1]["source"] == "cli"
        assert fresh._conversation_id == "bound-conversation"
        assert fresh.context_length == 131072


def test_sequence_rotation_cold_row_stays_in_child(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    original = harness.reopen

    def reopen_with_cold_row(engine: Any, root: Path, engines: ExitStack) -> Any:
        fresh = original(engine, root, engines)
        fresh._ingest_messages([{"role": "user", "content": "cold-phase row"}])
        return fresh

    monkeypatch.setattr(harness, "reopen", reopen_with_cold_row)
    payloads: list[str] = []
    observe(next(case for case in SCENARIOS if case.name == "sequence-rotation"), tmp_path, payloads=payloads)
    state = json.loads(next(event.removeprefix("cold-state ") for event in payloads
                            if event.startswith("cold-state ")))
    rows = [row for row in state["tables"]["messages"] if row[5] == "cold-phase row"]
    assert len(rows) == 1
    assert rows[0][1] == "child"
    assert state["_session_id"] == "child"
