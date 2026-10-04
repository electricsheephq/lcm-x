"""#823: B2 pairs parts through the host's own merge record, within one lineage."""
import json
import sqlite3
from pathlib import Path

import pytest

from bench.instruments.reliability.scorers import host_parity, multiset


A, B = "[T14] user turn 14: alpha", "[T15] user turn 15: beta"
TREE = str(Path(__file__).resolve().parents[1])


def host(tmp_path, *, uid_columns=True, active=1, first=A, second=B, extra=(), absorbed=None):
    db = tmp_path / "state.db"
    con = sqlite3.connect(db)
    columns = ", message_uid text, absorbed_message_uids text" if uid_columns else ""
    con.execute("create table messages (id integer, session_id text, role text, content text, active integer" + columns + ")")
    rows = [(38, "parent", "user", first, 0, "a", "[]"),
            (48, "child", "user", A + "\n\n" + B, active, "a", json.dumps(absorbed or ["b"])),
            (49, "child", "user", second, 1, "b", "[]"), *extra]
    con.executemany("insert into messages values (" + ",".join("?" * (7 if uid_columns else 5)) + ")",
                    [r if uid_columns else r[:5] for r in rows])
    con.commit()
    con.close()
    data, why = host_parity.load(db, lambda sid: "chat" if sid in ("parent", "child") else sid, TREE)
    assert why is None
    return data["chat"]["keys"]


def score(h, *, expected=None, texts=(A, B)):
    return multiset.score(expected or [("user", A + "\n\n" + B)],
                          [(i, "chat", "user", text) for i, text in enumerate(texts, 1)], h)


def test_host_merge_provenance_pairs_38_48_49(tmp_path):
    out = score(host(tmp_path))
    assert out["verdict"] == "PASS"
    assert out["host_parity_licensed"] == []
    assert out["missing_keys"] == out["surplus_rows"] == out["stored_rows_not_expected"] == 0
    [pair] = out["held_composites_as_parts"]
    assert pair["provenance"] == "host_merge_record" and pair["host_row_id"] == 48
    assert pair["as_parts"] == 1
    assert pair["parts"] == [{"sha256": multiset.h(A), "uses": 1, "store_ids": [1]},
                             {"sha256": multiset.h(B), "uses": 1, "store_ids": [2]}]
    assert "preview" not in pair


@pytest.mark.parametrize("case", ["no_columns", "wrong_join", "ambiguous_uid", "missing_stored",
                                  "inactive", "missing_first", "other_lineage", "too_many_parts", "carrier"])
def test_host_merge_provenance_guards_keep_failure(tmp_path, case):
    kw, texts = {}, (A, B)
    if case == "no_columns":
        kw["uid_columns"] = False
    elif case == "wrong_join":
        kw["second"] = "changed beta"
    elif case == "ambiguous_uid":
        kw["extra"] = [(50, "parent", "user", "other alpha", 0, "a", "[]")]
    elif case == "missing_stored":
        texts = (A,)
    elif case == "inactive":
        kw["active"] = 0
    elif case == "missing_first":
        kw["first"] = ""
    elif case == "other_lineage":
        kw["second"] = ""
        kw["extra"] = [(50, "cron", "user", B, 1, "b", "[]")]
    elif case == "too_many_parts":
        kw["absorbed"] = ["b"] * multiset.COVER_PARTS
    elif case == "carrier":
        from bench.instruments.reliability import plugin_tree
        header, prefixes = plugin_tree.carrier_markers(Path(TREE))
        assert prefixes
        kw["first"] = prefixes[0] + A
        assert header.match(kw["first"]) or kw["first"].startswith(prefixes)
    out = score(host(tmp_path, **kw), texts=texts)
    assert out["verdict"] == "FAIL" and out["missing_keys"] == 1
    assert out["held_composites_as_parts"] == []


def test_host_merge_provenance_reserves_expected_rows_and_consumes_only_parts(tmp_path):
    h = host(tmp_path)
    expected = [("user", A), ("user", A + "\n\n" + B)]
    assert score(h, expected=expected)["verdict"] == "FAIL"
    out = score(h, expected=expected, texts=(A, A, B, "unrelated surplus"))
    assert out["missing_keys"] == 0 and out["surplus_rows"] == 1 and out["verdict"] == "FAIL"
    [pair] = out["held_composites_as_parts"]
    assert [p["store_ids"] for p in pair["parts"]] == [[2], [3]]
    assert out["host_parity_licensed"] == []


def test_host_merge_provenance_same_uid_same_key_is_unambiguous(tmp_path):
    h = host(tmp_path, extra=[(50, "parent", "user", A, 1, "a", "[]")])
    assert score(h)["verdict"] == "PASS"


def test_host_merge_provenance_one_record_cannot_cover_two_deficits(tmp_path):
    out = score(host(tmp_path), expected=[("user", A + "\n\n" + B)] * 2, texts=(A, A, B, B))
    assert out["verdict"] == "FAIL" and out["deficit_rows"] == 1
    assert sum(p["as_parts"] for p in out["held_composites_as_parts"]) == 1


def test_host_merge_provenance_copied_record_cannot_cover_two_deficits(tmp_path):
    composite = A + "\n\n" + B
    h = host(tmp_path, extra=[(47, "parent", "user", composite, 1, "a", '["b"]')])
    out = score(h, expected=[("user", composite)] * 2, texts=(A, A, B, B))
    assert out["verdict"] == "FAIL" and out["deficit_rows"] == 1
    [pair] = out["held_composites_as_parts"]
    assert pair["as_parts"] == 1 and pair["host_row_id"] == 47


@pytest.mark.parametrize("absorbed,parts", [(["b", "b"], (A, B, B)), (["a", "b"], (A, A, B))])
def test_host_merge_provenance_repeated_uid_cannot_pair(tmp_path, absorbed, parts):
    composite = "\n\n".join(parts)
    h = host(tmp_path, extra=[(50, "child", "user", composite, 1, "a", json.dumps(absorbed))])
    out = score(h, expected=[("user", composite)], texts=parts)
    assert out["verdict"] == "FAIL" and out["deficit_rows"] == 1
    assert out["held_composites_as_parts"] == []


@pytest.mark.parametrize("texts", [(A, A, B), (A, B, B)])
def test_host_merge_provenance_constituent_rows_license_no_duplicate(tmp_path, texts):
    out = score(host(tmp_path), texts=texts)  # the host rows proving record 48 cannot also license a surplus part
    assert out["verdict"] == "FAIL" and out["surplus_rows"] == out["stored_rows_not_expected"] == 1
    assert out["host_parity_licensed"] == [] and out["held_composites_as_parts"][0]["host_row_id"] == 48


def test_host_merge_provenance_licenses_host_capacity_beyond_the_record(tmp_path):
    double = [(50, "parent", "user", A, 1, "a", "[]"), (51, "parent", "user", A, 1, "a", "[]")]
    out = score(host(tmp_path, extra=double), texts=(A, A, B))  # host view holds A twice: one spare licence
    assert out["verdict"] == "PASS" and [r["licensed"] for r in out["host_parity_licensed"]] == [1]
