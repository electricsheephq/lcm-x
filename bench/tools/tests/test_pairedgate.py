from __future__ import annotations

import hashlib
import json

import pytest

from bench.tools import pairedgate


def _gate() -> dict:
    return {
        "name": "scale-gate-v1",
        "instrument": "scale389",
        "primary_metric": "answer_turn_delivered_complete",
        "b_pool": {"count": 24},
        "pre_registration_timestamp": "2026-07-30T00:00:00Z",
        "bars": {
            "PASS": {
                "net": {"min": 8, "rungs": ["8000", "19829"]},
                "token_ratio": {"max": 1.6, "rungs": ["8000", "19829"]},
            },
            "GRAY": {
                "net": {"min": 3, "max": 7, "rungs": ["8000", "19829"]}
            },
            "KILL": {
                "net": {"max": 2, "rungs": ["8000", "19829"]},
                "token_ratio": {
                    "min": 2.2,
                    "scope": "any_passing_rung",
                },
            },
        },
    }


def test_registration_freezes_and_refuses_duplicate_name(tmp_path):
    gate_path = tmp_path / "gate.yaml"
    registry = tmp_path / "registry.jsonl"
    gate_path.write_text(json.dumps(_gate()), encoding="utf-8")

    entry = pairedgate.register_gate(gate_path, registry)

    assert entry["sha256"] == hashlib.sha256(gate_path.read_bytes()).hexdigest()
    assert len(registry.read_text(encoding="utf-8").splitlines()) == 1
    with pytest.raises(ValueError, match="already registered"):
        pairedgate.register_gate(gate_path, registry)


def test_power_refuses_pool_below_floor(tmp_path):
    gate_path = tmp_path / "gate.yaml"
    registry = tmp_path / "registry.jsonl"
    gate_path.write_text(json.dumps(_gate()), encoding="utf-8")
    pairedgate.register_gate(gate_path, registry)

    memo = pairedgate.power_gate("scale-gate-v1", registry, b_pool=12, floor=20, pass_net=8)

    assert memo["verdict"] == "UNDERPOWERED"
    assert memo["refused"] is True
    assert memo["floor_check"] == {"observed": 12, "required": 20, "ok": False}
    assert memo["pass_bar"]["required_conversion_rate"] == 8 / 12
    assert pairedgate.main(
        ["power", "scale-gate-v1", "--registry", str(registry), "--b-pool", "12"]
    ) == 3


def test_f45_shape_reads_ambiguous_cell_and_shows_arithmetic(tmp_path):
    gate_path = tmp_path / "gate.yaml"
    registry = tmp_path / "registry.jsonl"
    inputs = tmp_path / "GATE-INPUTS.txt"
    gate_path.write_text(json.dumps(_gate()), encoding="utf-8")
    inputs.write_text(
        "rung | b | c | net | median tok OFF | median tok ON | ratio\n"
        "500 | 18 | 0 | +18 | 1751.5 | 10006.5 | 5.71x\n"
        "2000 | 12 | 0 | +12 | 1741.5 | 9856.5 | 5.66x\n"
        "8000 | 3 | 0 | +3 | 1741.0 | 10699.5 | 6.15x\n"
        "19829 | 6 | 0 | +6 | 1717.5 | 10513.5 | 6.12x\n",
        encoding="utf-8",
    )
    pairedgate.register_gate(gate_path, registry)

    result = pairedgate.read_verdict("scale-gate-v1", inputs, registry)

    assert result["verdict"] == "AMBIGUOUS"
    assert "GRAY.net" in result["conflicts"]
    assert "KILL.token_ratio" in result["conflicts"]
    assert result["rungs"][-1]["net_arithmetic"] == "6 - 0 = 6"


def test_verify_detects_post_registration_edit(tmp_path):
    gate_path = tmp_path / "gate.yaml"
    registry = tmp_path / "registry.jsonl"
    gate_path.write_text(json.dumps(_gate()), encoding="utf-8")
    pairedgate.register_gate(gate_path, registry)

    gate_path.write_text(json.dumps({**_gate(), "primary_metric": "changed"}), encoding="utf-8")

    result = pairedgate.verify_registration("scale-gate-v1", registry)

    assert result["ok"] is False
    assert result["status"] == "FAIL"

