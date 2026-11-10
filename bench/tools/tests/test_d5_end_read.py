"""The end read's coverage check, verdict rule and pin refusals (synthetic rows; identity is the row's role and text)."""
from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

from bench.tools import d5_end_read as er


def identity(row):
    return row.get("role"), row.get("content")


def host(i, session, content, role="user"):
    return {"id": i, "session_id": session, "role": role, "content": content}


def stored(i, session, content, role="user"):
    return {"store_id": i, "session_id": session, "role": role, "content": content}


START = {"host": [host(1, "s1", "old")], "store": [stored(10, "s1", "old")]}
PRESERVED = {"verdict": "PASS"}


def read(end, end_dir="/nonexistent"):
    check2 = er.coverage(START, end, identity, end_dir)
    return check2, er.verdict(PRESERVED, check2)


def test_every_window_host_row_stored_in_order_passes():
    end = {"host": [*START["host"], host(2, "s1", "a"), host(3, "s2", "b")],
           "store": [*START["store"], stored(11, "s1", "a"), stored(12, "s2", "b")]}
    check2, verdict = read(end)
    assert verdict == "PASS"
    assert (check2["window_host_rows"], check2["matched"], check2["unmatched"], check2["lost"]) == (2, 2, 0, 0)


def test_rows_at_or_below_the_start_capture_are_not_in_the_window():
    end = {"host": [*START["host"]], "store": [*START["store"]]}
    check2, verdict = read(end)
    assert (check2["window_host_rows"], check2["new_stored_rows"], verdict) == (0, 0, "PASS")


def test_lost_only_when_the_session_holds_no_stored_row_at_all():
    end = {"host": [*START["host"], host(2, "s9", "never stored")], "store": [*START["store"]]}
    check2, verdict = read(end)
    assert (check2["lost"], check2["unmatched"], verdict) == (1, 0, "FAIL")
    assert check2["unmatched_rows"] == [{"host_id": 2, "session_sha256_12": er._sha12("s9"), "class": "LOST"}]


def test_a_row_the_host_rewrote_is_unmatched_with_a_hint_never_lost():
    end = {"host": [*START["host"], host(2, "s1", "  padded  ")], "store": [*START["store"], stored(11, "s1", "padded")]}
    check2, verdict = read(end)
    assert (check2["lost"], check2["unmatched"], verdict) == (0, 1, "INCONCLUSIVE")
    assert check2["unmatched_by_hint"]["equal after stripping edge whitespace"] == 1
    assert check2["unmatched_rows"][0]["candidate_store_id"] == 11


def test_a_payload_swapped_for_its_reference_gets_its_hint(tmp_path):
    (tmp_path / "lcm-large-outputs").mkdir()
    (tmp_path / "lcm-large-outputs" / "p1.json").write_text(json.dumps({"content": "the full tool output"}))
    end = {"host": [*START["host"], host(2, "s1", "the full tool output", role="tool")],
           "store": [*START["store"], stored(11, "s1", "[Externalized tool output: 20 chars; ref=p1.json]", role="tool")]}
    check2, verdict = read(end, tmp_path)
    assert check2["unmatched_by_hint"]["payload swapped for a reference"] == 1 and verdict == "INCONCLUSIVE"


def test_an_unexplained_difference_is_no_hint_and_inconclusive():
    end = {"host": [*START["host"], host(2, "s1", "one text")], "store": [*START["store"], stored(11, "s1", "another")]}
    check2, verdict = read(end)
    assert check2["unmatched_by_hint"] == {"equal after stripping edge whitespace": 0,
                                           "payload swapped for a reference": 0, "no hint": 1}
    assert (check2["stored_rows_not_matched"], verdict) == (1, "INCONCLUSIVE")


def test_check1_decides_first():
    clean = {"lost": 0, "unmatched": 0}
    assert er.verdict({"verdict": "FAIL"}, clean) == "FAIL"
    assert er.verdict({"verdict": "INCONCLUSIVE"}, clean) == "INCONCLUSIVE"
    assert er.verdict({"verdict": "PASS"}, {"lost": 1, "unmatched": 3}) == "FAIL"


def test_the_output_carries_no_row_text_and_names_sessions_by_hash():
    end = {"host": [*START["host"], host(2, "secret-session", "private words")],
           "store": [*START["store"], stored(11, "secret-session", "other private words")]}
    text = json.dumps(er.coverage(START, end, identity, "/nonexistent"))
    assert "private words" not in text and "secret-session" not in text
    assert er._sha12("secret-session") in text


def test_a_pin_is_refused_unless_it_hashes_to_the_sha256_sent(tmp_path):
    pin = tmp_path / "pin.json"
    pin.write_text("{}")
    good = hashlib.sha256(pin.read_bytes()).hexdigest()
    assert er._capture(pin, tmp_path, "XYZ", tmp_path)[1] == "the pin sha256 is not 64 lowercase hex"
    assert er._capture(pin, tmp_path, "0" * 64, tmp_path)[1] == "the pin file does not hash to the sha256 sent"
    assert er._capture(pin, tmp_path, good, tmp_path / "s")[1] is not None  # An empty pin never verifies.


def test_a_capture_that_does_not_verify_stops_the_read_before_the_kit_or_any_store(tmp_path):
    pin = tmp_path / "pin.json"
    pin.write_text("{}")
    args = SimpleNamespace(start=str(tmp_path), start_pin=str(pin), start_pin_sha256="0" * 64, end=str(tmp_path),
                           end_pin=str(pin), end_pin_sha256="0" * 64, scratch=str(tmp_path / "s"),
                           config="/nonexistent", instrument_dir="/nonexistent")
    result = er.end_read(args)
    assert result["verdict"] == "INCONCLUSIVE" and result["reason"].startswith("start: ")
    assert result["verdict_rule"] == er.VERDICT_RULE


def test_the_read_wires_p1_identity_rows_and_both_checks(monkeypatch, tmp_path):
    """The whole read with the runner's host faked: P1 must pass before identity or any store is read."""
    end = {"host": [*START["host"], host(2, "s1", "a")], "store": [*START["store"], stored(11, "s1", "a")]}
    seen = []

    class Host:
        def __init__(self, config, scratch):
            seen.append(("config", sorted(config)))

        def pins(self):
            seen.append("pins")
            return {"verdict": verdict_p1}

        def rules(self):
            return {"identity": identity}

        def rows(self, folder):
            return START if folder == "start-dir" else end

    monkeypatch.setattr(er, "_capture", lambda *a: ({"verdict": "PASS"}, None))
    monkeypatch.setattr(er.runner, "LocalHost", Host)
    (tmp_path / "decl.json").write_text(json.dumps({"manifest": "m", "rules_module": "r"}))
    args = SimpleNamespace(start="start-dir", start_pin="p", start_pin_sha256="a" * 64, end="end-dir", end_pin="p",
                           end_pin_sha256="b" * 64, scratch=str(tmp_path), config=str(tmp_path / "decl.json"),
                           instrument_dir="kit")
    verdict_p1 = "PASS"
    result = er.end_read(args)
    assert result["verdict"] == "PASS" and result["check1_preservation"]["verdict"] == "PASS"
    assert result["check2_coverage"]["matched"] == 1 and result["pins_sha256"] == {"start": "a" * 64, "end": "b" * 64}
    assert seen[0] == ("config", ["instrument_dir", "manifest", "rules_module"])
    verdict_p1 = "FAIL"
    assert er.end_read(args)["reason"] == "P1: the kit does not verify against its manifest"
