"""Backfill's frozen observable protocol and independent observer guard probes."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tests.behaviour import backfill_trace_harness as h


def values(payloads: list[str], label: str) -> list[Any]:
    return [json.loads(event[len(label) + 1:]) for event in payloads if event.startswith(label + " ")]


def compare(expected: list[str], actual: list[str]) -> None:
    difference = h.first_difference(expected, actual)
    assert difference is None, f"{difference}\n{h.TRACE_GUIDANCE}"


def reached(case: h.Scenario, payloads: list[str]) -> None:
    """Reject an accidental setup-error golden before treating a shape as recorded."""
    if case.shape.endswith("non-sqlite-error"):
        message = "scripted open failure" if case.shape.startswith("read-open") else "scripted read failure"
        assert values(payloads, "runner raises") == [["RuntimeError", message, None]]
        if not case.shape.startswith("read-open"):
            assert values(payloads, "connection close")[0] == "connection-1"
            assert next(i for i, event in enumerate(payloads) if event.startswith("connection close ")) < next(
                i for i, event in enumerate(payloads) if event.startswith("runner raises "))
        return
    assert not values(payloads, "runner raises")
    reports = values(payloads, "report fields")
    assert reports
    first = reports[0]
    shape = case.shape
    refusals = {"disabled", "db-open-error", "profile-query-error", "dtype-mismatch", "lease-held",
                "profile-absent-apply", "cloud-apply-consent-off"}
    if shape == "profile-absent-dry" and case.runner == "summary":
        refusals.add(shape)
    if shape in refusals:
        assert isinstance(first, str)
        assert "error:" in first
        if shape in {"disabled", "db-open-error", "dtype-mismatch", "cloud-apply-consent-off", "profile-absent-apply"}:
            assert "status: refused" in first
        if shape == "lease-held":
            assert values(payloads, "return type")[0] == ["_BackfillRefusal", "lease_held"]
        if shape == "cloud-apply-consent-off":
            assert not any(event.startswith(("store ", "provider ")) for event in payloads)
        return
    assert isinstance(first, list)
    report = dict(first)
    assert "'function' object" not in str(report)
    if shape == "dtype-match" or shape.startswith("dry-") or shape.startswith("cloud-dry-") or shape == "profile-absent-dry":
        assert report["status"] == "dry-run"
        assert not values(payloads, "provider dispatch")
        if shape == "dry-token-cap":
            assert report["estimated_cost_tokens"] < report["estimated_tokens"]
    if shape in {"apply-empty", "apply-one", "page-B-minus-1", "page-B", "page-B-plus-1", "cloud-apply-consent-on"}:
        assert report["status"] == "complete"
        assert report["embedded"] == {"apply-empty": 0, "apply-one": 1, "page-B-minus-1": 2, "page-B": 3, "page-B-plus-1": 4}.get(shape, 3)
    if shape.startswith("provider-m"):
        assert report["status"] == "error" and report["embedded"] == 0
    stops = {"lease-renew-lost-after-page": "lease_lost", "budget-after-page": "op_budget_exhausted",
             "stop-before-dispatch": "process_exit", "stop-after-accepted-subbatch": "process_exit",
             "publication-owner-loss": "lease_lost", "publication-identity-superseded": "identity_superseded"}
    if shape in stops:
        assert report["stop_reason"] == stops[shape]
    if shape == "provider-auth-rejection":
        assert "authentication failed" in report["error"]
    if shape == "provider-ambiguous-after-subbatch":
        assert report["embedded"] > 0 and report["uncertain"] == 1
        assert dict(reports[-1])["embedded"] == 1
    if shape.startswith("publish-"):
        assert any(reason.startswith("local_error:" if shape == "publish-call-error" else "record_error:")
                   for _, reason in report["failed"])
    if shape == "release-failure":
        assert report["error"].startswith("lease release failed:")
    if shape in {"invalid-accepted-index", "invalid-dispatch-index", "unresolved-request"}:
        assert report["failed"] or report["uncertain"] or report["status"] == "partial"
        assert report["embedded"] == 0
    if shape == "identity-switch-before-dispatch":
        race = next(i for i, event in enumerate(payloads) if event.startswith("race "))
        if case.runner == "summary":
            assert not values(payloads, "provider dispatch")
            assert "active embedding identity changed before backfill dispatch" in report["error"]
        else:
            dispatch = next(i for i, event in enumerate(payloads) if event.startswith("provider dispatch "))
            assert race < dispatch
            assert report["stop_reason"] == "identity_superseded"
            # C6 rebuilds identity from the same read-profile tuple; this race
            # reaches publication protection, not that tautological guard.
    if shape.startswith("incremental-"):
        assert report["stop_reason"] == "process_exit"
        assert values(payloads, "reopen")
        assert dict(reports[1])["embedded"] > 0
        if shape == "incremental-revision-gap-checkpoint":
            state = values(payloads, "discovery state")[0]
            assert any(key.startswith("chunk_gapfill_scan:") for key, _ in state["tables"]["metadata"])
    if shape == "contextual-model-cap-and-estimates":
        preview, applied = (dict(report) for report in reports)
        assert preview["estimated_batches"] == len(values(payloads, "provider dispatch"))
        assert applied["skipped"] and applied["embedded"] == 4
        assert preview["estimated_cost_tokens"] == applied["consumed_tokens"]


@pytest.mark.parametrize("case", h.SCENARIOS, ids=lambda case: case.name)
def test_backfill_golden(case: h.Scenario, tmp_path: Path) -> None:
    expected: dict[str, list[str]] = json.loads(h.GOLDEN.read_text())
    payloads: list[str] = []
    actual = h.observe(case, tmp_path, payloads=payloads)
    if h.first_difference(expected[case.name], actual) is not None:
        (tmp_path / "actual-payload.json").write_text(json.dumps(payloads, ensure_ascii=False))
    reached(case, payloads)
    compare(expected[case.name], actual)


def test_fixture_inventory_and_size() -> None:
    data = json.loads(h.GOLDEN.read_text())
    assert list(data) == [case.name for case in h.SCENARIOS]
    assert len(data) == len(set(data)) == 80
    payloads = [json.dumps(trace) for trace in data.values()]
    assert len(payloads) == len(set(payloads)), "duplicate scenario payloads"
    assert h.GOLDEN.stat().st_size < 5_000_000


@pytest.mark.parametrize("probe", ["primitive", "table", "provider"])
def test_observer_completeness(probe: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    if probe == "provider":
        monkeypatch.setattr(h.ScriptedProvider, "trace_probe", lambda self: None, raising=False)
        original_resolve = h.ScriptedProvider.embed_document_batches
        def unknown_provider(self: Any, *args: Any, **kwargs: Any) -> Any:
            self.trace_probe()
            return original_resolve(self, *args, **kwargs)
        monkeypatch.setattr(h.ScriptedProvider, "embed_document_batches", unknown_provider)
        with pytest.raises(AssertionError, match="unregistered provider method"):
            h.observe(h.Scenario("summary", "apply-one"), tmp_path)
        return
    original = h.vectors.VectorStore._init_db
    def unknown(self: Any) -> None:
        original(self)
        if probe == "primitive":
            self.trace_probe()
        else:
            self.connection.execute("CREATE TABLE trace_probe(value TEXT)")
    monkeypatch.setattr(h.vectors.VectorStore, "trace_probe", lambda self: None, raising=False)
    monkeypatch.setattr(h.vectors.VectorStore, "_init_db", unknown)
    with pytest.raises(AssertionError, match="unregistered store primitive" if probe == "primitive" else "unregistered table write"):
        h.observe(h.Scenario("chunks", "apply-one"), tmp_path)


def test_cloud_consent_dry_previews_match(tmp_path: Path) -> None:
    previews = []
    for suffix in ("off", "on"):
        payloads: list[str] = []
        h.observe(h.Scenario("chunks", "cloud-dry-consent-" + suffix), tmp_path / suffix, payloads=payloads)
        previews.append(values(payloads, "rendered"))
    assert previews[0] == previews[1]


def test_normalization_preserves_content_order_and_errors(tmp_path: Path) -> None:
    observer = h.Observer(tmp_path)
    text = 'status: partial\n"café"\\\t00000000000000000000000000000001'
    assert observer.value(text) == text
    assert list(observer.value({"second": 2, "first": 1})) == ["second", "first"]
    assert observer.value(RuntimeError("exact error")) == ["RuntimeError", "exact error", None]
    row = ["synthetic"]
    assert observer.alias(row) == observer.alias(row)
    assert observer.alias(row) != observer.alias(list(row))


def perturb(kind: str, events: list[str], payloads: list[str]) -> list[str]:
    changed = list(events)
    if kind == "write":
        index = next(i for i, event in enumerate(payloads) if event.startswith("sql ") and "INSERT INTO lcm_chunk_meta" in event)
        del changed[index]
    elif kind == "order":
        index = next(i for i, event in enumerate(payloads) if event.startswith("provider accepted "))
        changed[index], changed[index + 1] = changed[index + 1], changed[index]
    else:
        label = "report fields" if kind == "report" else "accepted checkpoint"
        index = next(i for i, event in enumerate(payloads) if event.startswith(label + " "))
        value = json.loads(payloads[index][len(label) + 1:])
        if kind == "report":
            value[0][1] = "perturbed mode"
        else:
            key = next(iter(value["cursors"]))
            value["cursors"][key][1] += 1
        observer = h.Observer(Path("__unused_trace_root__"))
        observer.emit(label, value)
        changed[index] = observer.events[0]
    return changed


@pytest.mark.parametrize("kind", ["report", "write", "order", "cursor"])
def test_trace_sensitivity(kind: str, tmp_path: Path) -> None:
    payloads: list[str] = []
    actual = h.observe(h.Scenario("chunks", "apply-one"), tmp_path, payloads=payloads)
    with pytest.raises(AssertionError, match="first divergence"):
        compare(actual, perturb(kind, actual, payloads))
