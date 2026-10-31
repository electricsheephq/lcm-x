"""Fast, engine-free checks for the occurrence fuzz generator and scoring bar."""
from __future__ import annotations

import hashlib
import json
import types
from pathlib import Path
from typing import Any

import pytest

from bench.instruments.occurrence_fuzz import driver, gate, gen


@pytest.mark.parametrize("mode,expected", [
    ("base", "8566d2e09208abdf91fd6e4c62eb9081170a7e3cdc1f7dc8e1cfbdddd4ccd581"),
    ("ep", "58e263c8b64286d3c48222bd81d0bd79a025707614cfff6b2da9dd756e36f250"),
])
def test_generator_hash(mode: str, expected: str) -> None:
    data = "".join(json.dumps(gen.gen(seed, mode)) + "\n" for seed in range(100))
    assert hashlib.sha256(data.encode()).hexdigest() == expected


@pytest.mark.parametrize("mode", ["base", "ep"])
def test_lifecycle_ops_are_labelled(mode: str) -> None:
    sequences = [gen.gen(seed, mode) for seed in range(100)]
    for operation in ("reset", "restart"):
        labelled = [seq for seq in sequences if operation in seq["ops"]]
        assert labelled
        assert all(seq["views"].count(operation.upper()) == seq["ops"].count(operation)
                   for seq in sequences)


def fixture_records(ops: list[str]) -> tuple[dict[int, gate.Record], dict[int, gate.Record]]:
    seqs = {1: {"seed": 1, "masks": "n", "ops": ops, "rows": {
        "B0": {"role": "user", "content": "request"},
        "B1": {"role": "assistant", "content": "reply"},
    }}}
    reference = {1: {"seed": 1, "ledger": {"user|request": 1, "assistant|reply": 1}, "err": None}}
    return seqs, reference


def test_gate_self_comparison() -> None:
    seqs, reference = fixture_records([])
    totals, errors = gate.score(seqs, reference, [("base", reference), ("candidate1", reference)])
    assert errors == 0
    assert all(totals["none"][f"candidate1_{metric}"] == 0 for metric in ("loss", "dup", "dup_nonresend"))
    assert gate.report(totals, errors, ["base", "candidate1"])


@pytest.mark.parametrize("ops,excused", [([], False), (["resend:B1"], False), (["resend:B0"], True)])
def test_gate_duplicate_needs_matching_resend(ops: list[str], excused: bool) -> None:
    seqs, reference = fixture_records(ops)
    candidate = {1: {"ledger": {"user|request": 2, "assistant|reply": 1}}}
    totals, errors = gate.score(seqs, reference, [("base", reference), ("candidate1", candidate)])
    assert totals["none"]["candidate1_dup"] == 1
    assert totals["none"]["candidate1_dup_nonresend"] == int(not excused)
    assert gate.report(totals, errors, ["base", "candidate1"]) is excused


def test_gate_loss() -> None:
    seqs, reference = fixture_records(["resend:B1"])
    candidate = {1: {"ledger": {"user|request": 1}}}
    totals, errors = gate.score(seqs, reference, [("base", reference), ("candidate1", candidate)])
    assert totals["none"]["candidate1_loss"] == 1
    assert not gate.report(totals, errors, ["base", "candidate1"])


@pytest.mark.parametrize("damage", ["errored", "missing"])
def test_gate_counts_reference_errors(damage: str) -> None:
    seqs, reference = fixture_records([])
    good = {1: dict(reference[1])}
    if damage == "errored":
        reference[1]["err"] = "RuntimeError"
    else:
        del reference[1]
    totals, errors = gate.score(seqs, reference, [("base", good), ("candidate1", good)])
    assert errors == 1
    assert not gate.report(totals, errors, ["base", "candidate1"])


def test_gate_fails_when_nothing_is_scored() -> None:
    totals, errors = gate.score({}, {}, [("base", {}), ("candidate1", {})])
    assert errors == 0
    assert not gate.report(totals, errors, ["base", "candidate1"])


class _ResetFails:
    def __init__(self, **_: Any) -> None:
        self._store = types.SimpleNamespace(get_session_messages=lambda _sid: [])

    def on_session_start(self, *_: Any, **__: Any) -> None:
        pass

    def on_session_reset(self) -> None:
        raise RuntimeError("reset failed")

    def ingest(self, _messages: Any) -> None:
        pass

    def shutdown(self) -> None:
        pass


def test_driver_records_lifecycle_errors(tmp_path: Path) -> None:
    config_module = types.SimpleNamespace(LCMConfig=types.SimpleNamespace)
    engine_module = types.SimpleNamespace(LCMEngine=_ResetFails)
    seq = {"seed": 7, "rows": {"B0": {"role": "user", "content": "request"}},
           "views": [["B0"], "RESET", ["B0"]]}
    result = driver.run(seq, tmp_path / "run", config_module, engine_module, driver._Catch())
    assert result["err"] == "RuntimeError"
    assert result["counts"] == [0]


def test_gate_multiple_candidates_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                      capsys: pytest.CaptureFixture[str]) -> None:
    seqs, reference = fixture_records([])
    paths = []
    for name, rows in (("seqs", seqs), ("reference", reference)):
        path = tmp_path / f"{name}.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows.values()))
        paths.append(str(path))
    monkeypatch.setattr("sys.argv", ["gate.py", *paths, paths[1], paths[1], paths[1]])
    gate.main()
    output = capsys.readouterr().out
    assert "candidate2 0 candidate1 0 base 0" in output
    assert "errors 0 GATE PASS" in output
    # Earlier candidates are diagnostic; the last candidate sets the verdict.
    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps({"seed": 1, "ledger": {}}) + "\n")
    monkeypatch.setattr("sys.argv", ["gate.py", *paths, paths[1], str(bad), paths[1]])
    gate.main()
    output = capsys.readouterr().out
    assert "LOSS candidate2 0 candidate1 2 base 0" in output
    assert "errors 0 GATE PASS" in output
    assert callable(driver.run)


def test_gate_reference_multiplicity_and_classes() -> None:
    seqs, reference = fixture_records([])
    reference[1]["ledger"]["user|request"] = 2
    candidate: dict[int, dict[str, Any]] = {1: {"ledger": {"user|request": 2, "assistant|reply": 1}}}
    totals, _ = gate.score(seqs, reference, [("candidate1", candidate)])
    assert totals["none"]["candidate1_dup"] == 0
    assert [gate.cls(mask) for mask in ("nn", "ss", "uu", "nsu")] == ["none", "stamp", "uid", "mixed"]
