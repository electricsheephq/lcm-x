"""Hermetic B1/B2 externalized-row regression, using the existing scorer fixture unchanged."""
from __future__ import annotations

import base64
import os
import json

import pytest

import externalize
from tests import test_reliability_scorers as fixture
from bench.instruments.reliability.scorers import bars, multiset


def externalized_cell(tmp_path, *, role="user", failure=None):
    events, rows, kw = fixture.clean_events(), fixture.clean_rows(), {}
    index = 0 if role == "user" else 1
    if role == "tool":
        events, rows, kw = fixture.tool_cell()
        index = next(i for i, r in enumerate(rows) if r[0] == "tool")
    original = rows[index][1]
    home = tmp_path / "cell" / "hermes-home"
    storage = home / externalize.DEFAULT_LARGE_OUTPUT_DIRNAME
    storage.mkdir(parents=True)
    ref = "fixture-payload.json"
    payload = {"kind": "tool_result" if role == "tool" else "raw_payload", "role": role,
               "content": original, "content_chars": len(original), "content_bytes": len(original.encode())}
    if failure == "corrupted":
        payload["content"] = original[:-1] + "!"
    elif failure == "no_content":
        del payload["content"]
    elif failure == "invalid_content":
        payload["content"] = None
    path = storage / ref
    if failure != "missing":
        path.write_text("{" if failure == "json" else json.dumps(payload))
    stub = externalize._build_externalized_placeholder({**payload, "ref": ref})
    rows[index] = (role, stub, *rows[index][2:])
    out = fixture.make(tmp_path, rows=rows, events=events, **kw)
    return out, tmp_path / "cell", original, index + 1


@pytest.mark.parametrize("role", ["user", "assistant", "tool"])
def test_resolved_payload_matches_raw_key(tmp_path, role):
    out, d, original, store_id = externalized_cell(tmp_path, role=role)
    assert out["verdict"] == "PASS"
    assert "B1" not in out["failed_bars"] and "B2" not in out["failed_bars"]
    b2 = out["numbers"]["B2"]
    assert b2["deficit_rows"] == b2["surplus_rows"] == b2["tool_missing_rows"] == b2["tool_surplus_rows"] == 0
    assert b2["externalized"] == {"resolved": 1, "unresolved": 0, "deficit_rows": 0,
                                   "reasons": {}, "store_ids": [], "deficits": [], "content_mismatch_rows": 0, "content_mismatch": []}
    from bench.instruments.reliability.scorers import externalized
    resolved, _, _ = externalized.resolve([(store_id, "S0", role,
        externalize._build_externalized_placeholder({"ref": "fixture-payload.json"}), None, None)], d / "hermes-home")
    assert multiset.h(resolved[0][3]) == multiset.h(original)


@pytest.mark.parametrize("role", ["user", "assistant", "tool"])
def test_one_byte_corruption_fails_b2_and_is_named(tmp_path, role):
    out, _, _, store_id = externalized_cell(tmp_path, role=role, failure="corrupted")
    assert out["verdict"] == "FAIL" and "B2" in out["failed_bars"]
    b2 = out["numbers"]["B2"]
    assert b2["deficit_rows"] + b2["tool_missing_rows"] == 1  # ordinary B2 accounting fails the wrong bytes
    ext = b2["externalized"]
    assert ext["resolved"] == 1 and ext["unresolved"] == 0 and ext["deficit_rows"] == 0 and ext["reasons"] == {}
    assert ext["content_mismatch_rows"] == 1 and [m["store_id"] for m in ext["content_mismatch"]] == [store_id]


@pytest.mark.parametrize("failure,reason", [("missing", "missing_payload_file"), ("json", "unreadable_payload"),
                                            ("no_content", "no_content_field"), ("invalid_content", "invalid_content")])
def test_unresolved_payload_fails_with_reason(tmp_path, failure, reason):
    out, _, _, store_id = externalized_cell(tmp_path, failure=failure)
    assert out["verdict"] == "FAIL" and "B2" in out["failed_bars"]
    ext = out["numbers"]["B2"]["externalized"]
    assert ext == {"resolved": 0, "unresolved": 1, "deficit_rows": 1, "reasons": {reason: 1},
                   "store_ids": [store_id], "deficits": [{"store_id": store_id, "reason": reason}], "content_mismatch_rows": 0, "content_mismatch": []}


def test_plain_existing_fixture_is_unchanged(tmp_path):
    out = fixture.make(tmp_path, rows=fixture.clean_rows(), events=fixture.clean_events())
    assert out["verdict"] == "PASS"
    assert out["numbers"]["B2"]["deficit_rows"] == out["numbers"]["B2"]["surplus_rows"] == 0
    # Drop only the newly added diagnostics; compare the entire result with the original row reader.
    ext = out["numbers"]["B2"].pop("externalized")
    assert ext == {"resolved": 0, "unresolved": 0, "deficit_rows": 0, "reasons": {}, "store_ids": [], "deficits": [],
                   "content_mismatch_rows": 0, "content_mismatch": []}
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(bars.externalized, "resolve", lambda rows, home: (rows, dict(ext), set()))
        plain = bars.score({"id": "t", "tool_plan": [], "native_recovery": False, "min_compactions": 2,
                            "final_compaction_check": True, "bars": list(fixture.cells.BARS)}, tmp_path / "cell")
    plain["numbers"]["B2"].pop("externalized")
    assert out == plain


def test_resolved_user_edges_keep_b2_normalization(tmp_path):
    out, d, _, _ = externalized_cell(tmp_path)
    path = d / "hermes-home" / externalize.DEFAULT_LARGE_OUTPUT_DIRNAME / "fixture-payload.json"
    payload = json.loads(path.read_text())
    payload["content"] = " \n" + payload["content"] + "\t "
    path.write_text(json.dumps(payload))
    cell = {"bars": ["B1", "B2"], "min_compactions": 2}
    scored = bars.score(cell, d)
    assert scored["verdict"] == out["verdict"] == "PASS"


def test_operator_base_dir_does_not_reach_the_cell(tmp_path, monkeypatch):
    elsewhere = tmp_path / "operator-base"
    monkeypatch.setenv("LCM_HERMES_BASE_DIR", str(elsewhere))
    out, _, _, _ = externalized_cell(tmp_path)
    assert out["verdict"] == "PASS" and out["numbers"]["B2"]["externalized"]["resolved"] == 1
    assert os.environ["LCM_HERMES_BASE_DIR"] == str(elsewhere)


def test_ingest_placeholder_resolves_and_missing_payload_is_a_deficit(tmp_path):
    from hermes_lcm import ingest_protection
    from hermes_lcm.config import LCMConfig
    from bench.instruments.reliability.scorers import externalized

    assert externalized.INGEST_RE.pattern == ingest_protection._INGEST_PLACEHOLDER_RE.pattern
    assert externalized.INGEST_PREFIX == ingest_protection._EXTERNALIZED_PLACEHOLDER_PREFIX
    protect_message_for_ingest = ingest_protection.protect_message_for_ingest

    home = tmp_path / "hermes-home"
    original = "see this: data:image/png;base64," + base64.b64encode(b"LCM ingest payload " * 900).decode()
    # The ingest-protection placeholder path: large-output externalization (default on, #1013 part C) off.
    stored = protect_message_for_ingest({"role": "user", "content": original},
                                        LCMConfig(large_output_externalization_enabled=False),
                                        hermes_home=str(home), session_id="S0")["content"]
    assert externalized.INGEST_PREFIX in stored and stored != original
    rows, numbers, ids = externalized.resolve([(7, "S0", "user", stored, None, None)], home)
    assert rows[0][3] == original and ids == {7} and numbers["resolved"] == 1 and numbers["deficit_rows"] == 0
    payload_path = next((home / externalize.DEFAULT_LARGE_OUTPUT_DIRNAME).iterdir())
    raw = json.loads(payload_path.read_text())
    for broken in ({k: v for k, v in raw.items() if k != "content"}, {**raw, "content": None}):
        payload_path.write_text(json.dumps(broken))
        rows, numbers, _ = externalized.resolve([(7, "S0", "user", stored, None, None)], home)
        assert rows == [] and numbers["reasons"] == {"missing_ingest_payload": 1}
    payload_path.unlink()
    rows, numbers, ids = externalized.resolve([(7, "S0", "user", stored, None, None)], home)
    assert rows == [] and numbers["reasons"] == {"missing_ingest_payload": 1} and numbers["deficit_rows"] == 1


def test_default_config_media_payload_placeholder_resolves(tmp_path):
    # #1013 part C: with the defaults, a user data URI is stored as a large-output media_payload
    # reference; the scorer must resolve it (otherwise the default would recreate #1056).
    from hermes_lcm import ingest_protection
    from hermes_lcm.config import LCMConfig
    from bench.instruments.reliability.scorers import externalized

    home = tmp_path / "hermes-home"
    original = "see this: data:image/png;base64," + base64.b64encode(b"LCM ingest payload " * 900).decode()
    stored = ingest_protection.protect_message_for_ingest({"role": "user", "content": original}, LCMConfig(),
                                                          hermes_home=str(home), session_id="S0")["content"]
    assert stored != original and "kind=media_payload" in stored
    rows, numbers, ids = externalized.resolve([(7, "S0", "user", stored, None, None)], home)
    assert rows[0][3] == original and ids == {7} and numbers["resolved"] == 1 and numbers["deficit_rows"] == 0


def test_mismatch_diagnostic_skips_unscored_roles_and_groups():
    from bench.instruments.reliability.scorers import externalized

    numbers = {"content_mismatch_rows": 0, "content_mismatch": []}
    per = {"G": ([("user", "hello")], [])}
    rows = [(1, "G", "system", "a resolved system prompt", None, None),
            (2, "OTHER", "user", "a resolved row from an unrelated session", None, None),
            (3, "G", "user", "hello", None, None)]
    externalized.mismatches(numbers, rows, {1, 2, 3}, per, lambda session: session, set(), set(), None)
    assert numbers == {"content_mismatch_rows": 0, "content_mismatch": []}


def test_kept_db_copy_keeps_its_payloads_for_a_rescore(tmp_path):
    from bench.instruments.reliability import run_matrix

    scratch, d = tmp_path / "scratch", tmp_path / "cell"
    (scratch / "db").mkdir(parents=True)
    (scratch / "db" / "lcm.db").write_text("db")
    payloads = scratch / "hermes-home" / externalize.DEFAULT_LARGE_OUTPUT_DIRNAME
    payloads.mkdir(parents=True)
    (payloads / "p.json").write_text("{}")
    (scratch / "hermes-home" / "config.yaml").write_text("x")
    d.mkdir()
    run_matrix.release_scratch(scratch, d, {"verdict": "PASS"}, "all", False)
    assert (d / "db" / "lcm.db").exists() and not scratch.exists()
    assert (d / "hermes-home" / externalize.DEFAULT_LARGE_OUTPUT_DIRNAME / "p.json").exists()
    assert not (d / "hermes-home" / "config.yaml").exists()
