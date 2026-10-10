"""Hermetic B1/B2 externalized-row regression, using the existing scorer fixture unchanged."""
from __future__ import annotations

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
                                   "reasons": {}, "store_ids": [], "deficits": []}
    from bench.instruments.reliability.scorers import externalized
    resolved, _, _ = externalized.resolve([(store_id, "S0", role,
        externalize._build_externalized_placeholder({"ref": "fixture-payload.json"}), None, None)], d / "hermes-home")
    assert multiset.h(resolved[0][3]) == multiset.h(original)


@pytest.mark.parametrize("role", ["user", "assistant", "tool"])
def test_one_byte_corruption_is_externalized_deficit(tmp_path, role):
    out, _, _, store_id = externalized_cell(tmp_path, role=role, failure="corrupted")
    assert out["verdict"] == "FAIL" and "B2" in out["failed_bars"]
    ext = out["numbers"]["B2"]["externalized"]
    assert ext["resolved"] == 1 and ext["unresolved"] == 0 and ext["deficit_rows"] == 1
    assert ext["reasons"] == {"content_mismatch": 1} and ext["store_ids"] == [store_id]


@pytest.mark.parametrize("failure,reason", [("missing", "missing_payload_file"), ("json", "unreadable_payload"),
                                            ("no_content", "no_content_field"), ("invalid_content", "invalid_content")])
def test_unresolved_payload_fails_with_reason(tmp_path, failure, reason):
    out, _, _, store_id = externalized_cell(tmp_path, failure=failure)
    assert out["verdict"] == "FAIL" and "B2" in out["failed_bars"]
    ext = out["numbers"]["B2"]["externalized"]
    assert ext == {"resolved": 0, "unresolved": 1, "deficit_rows": 1, "reasons": {reason: 1},
                   "store_ids": [store_id], "deficits": [{"store_id": store_id, "reason": reason}]}


def test_plain_existing_fixture_is_unchanged(tmp_path):
    out = fixture.make(tmp_path, rows=fixture.clean_rows(), events=fixture.clean_events())
    assert out["verdict"] == "PASS"
    assert out["numbers"]["B2"]["deficit_rows"] == out["numbers"]["B2"]["surplus_rows"] == 0
    # Drop only the newly added diagnostics; compare the entire result with the original row reader.
    ext = out["numbers"]["B2"].pop("externalized")
    assert ext == {"resolved": 0, "unresolved": 0, "deficit_rows": 0, "reasons": {}, "store_ids": [], "deficits": []}
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
