"""The end read's coverage check, verdict rule and pin refusals (synthetic rows; identity is the row's role and text)."""
from __future__ import annotations

import hashlib
import json
import sqlite3
from types import SimpleNamespace

from bench.tools import d5_end_read as er


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else "0" * 64


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
    assert (check2["window_host_rows"], check2["new_stored_rows"], verdict) == (0, 0, "INCONCLUSIVE")  # Proves nothing.


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
                                           "payload swapped for a reference": 0,
                                           "session never stored; a parent session matches a declared pattern": 0,
                                           "no hint": 1}
    assert (check2["stored_rows_not_matched"], verdict) == (1, "INCONCLUSIVE")


def test_check1_decides_first():
    clean = {"lost": 0, "unmatched": 0, "window_host_rows": 1}
    assert er.verdict({"verdict": "FAIL"}, clean) == "FAIL"
    assert er.verdict({"verdict": "INCONCLUSIVE"}, clean) == "INCONCLUSIVE"
    assert er.verdict({"verdict": "PASS"}, {"lost": 1, "unmatched": 3, "window_host_rows": 4}) == "FAIL"
    assert er.verdict({"verdict": "PASS"}, clean) == "PASS"


def test_a_session_whose_only_stored_rows_predate_the_window_is_unmatched_not_lost():
    end = {"host": [*START["host"], host(2, "s1", "new and not stored")], "store": [*START["store"]]}
    check2, verdict = read(end)
    assert (check2["lost"], check2["unmatched"], verdict) == (0, 1, "INCONCLUSIVE")


def test_matching_keeps_the_order_within_a_session():
    end = {"host": [*START["host"], host(2, "s1", "a"), host(3, "s1", "b")],
           "store": [*START["store"], stored(11, "s1", "b"), stored(12, "s1", "a")]}
    check2, _ = read(end)
    assert (check2["matched"], check2["unmatched"]) == (1, 1)


def test_a_host_side_reference_to_the_stored_payload_gets_the_payload_hint(tmp_path):
    (tmp_path / "lcm-large-outputs").mkdir()
    (tmp_path / "lcm-large-outputs" / "p2.json").write_text(json.dumps({"content": "stored in full"}))
    end = {"host": [*START["host"], host(2, "s1", "[Externalized tool output: 14 chars; ref=p2.json]", role="tool")],
           "store": [*START["store"], stored(11, "s1", "stored in full", role="tool")]}
    check2, _ = read(end, tmp_path)
    assert check2["unmatched_by_hint"]["payload swapped for a reference"] == 1


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


def test_a_pin_that_hashes_right_but_does_not_verify_is_refused(monkeypatch, tmp_path):
    pin = tmp_path / "pin.json"
    pin.write_text("{}")
    good = hashlib.sha256(pin.read_bytes()).hexdigest()
    monkeypatch.setattr(er.snapshot, "verify", lambda *a: {"verdict": "FAIL"})
    assert er._capture(pin, tmp_path, good, tmp_path / "s")[1] == "the capture does not verify against its pin"


def test_the_same_capture_twice_is_inconclusive():
    args = SimpleNamespace(start_pin_sha256="a" * 64, end_pin_sha256="a" * 64)
    assert er.end_read(args)["reason"] == "the start and end pins are the same capture"


def test_an_unexpected_error_prints_a_fixed_line_never_a_traceback(monkeypatch, capsys):
    monkeypatch.setattr(er, "end_read", lambda args: (_ for _ in ()).throw(FileNotFoundError("/private/typed/path")))
    code = er.main(["--config", "c", "--config-sha256", "3" * 64, "--instrument-dir", "k", "--scratch", "s", "--start", "a", "--start-pin", "p",
                    "--start-pin-sha256", "1" * 64, "--end", "b", "--end-pin", "q", "--end-pin-sha256", "2" * 64])
    out = capsys.readouterr()
    assert code == 2 and "typed" not in out.out + out.err and "unexpected FileNotFoundError" in out.out


def test_a_capture_that_does_not_verify_stops_the_read_before_the_kit_or_any_store(tmp_path):
    pin = tmp_path / "pin.json"
    pin.write_text("{}")
    args = SimpleNamespace(start=str(tmp_path), start_pin=str(pin), start_pin_sha256="0" * 64, end=str(tmp_path),
                           end_pin=str(pin), end_pin_sha256="1" * 64, scratch=str(tmp_path / "s"),
                           config="/nonexistent", config_sha256="0" * 64, instrument_dir="/nonexistent")
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
    lcm = {"ignore_session_patterns": [], "stateless_session_patterns": []}
    (tmp_path / "decl.json").write_text(json.dumps({"manifest": "m", "rules_module": "r", "settings": {"lcm": lcm}}))
    args = SimpleNamespace(start="start-dir", start_pin="p", start_pin_sha256="a" * 64, end="end-dir", end_pin="p",
                           end_pin_sha256="b" * 64, scratch=str(tmp_path), config=str(tmp_path / "decl.json"), config_sha256=_sha(tmp_path / "decl.json"),
                           instrument_dir="kit")
    verdict_p1 = "PASS"
    result = er.end_read(args)
    assert result["verdict"] == "PASS" and result["check1_preservation"]["verdict"] == "PASS"
    assert result["check2_coverage"]["matched"] == 1 and result["pins_sha256"] == {"start": "a" * 64, "end": "b" * 64}
    assert seen[0] == ("config", ["instrument_dir", "manifest", "rules_module", "settings"])
    verdict_p1 = "FAIL"
    assert er.end_read(args)["reason"] == "P1: the kit does not verify against its manifest"


def test_condition_1_a_declaration_without_the_session_patterns_is_inconclusive(monkeypatch, tmp_path):
    monkeypatch.setattr(er, "_capture", lambda *a: ({"verdict": "PASS"}, None))
    for decl in ({"manifest": "m"}, {"settings": {"lcm": {"ignore_session_patterns": []}}}):
        (tmp_path / "decl.json").write_text(json.dumps(decl))
        args = SimpleNamespace(start="a", start_pin="p", start_pin_sha256="a" * 64, end="b", end_pin="p",
                               end_pin_sha256="b" * 64, scratch=str(tmp_path), config=str(tmp_path / "decl.json"), config_sha256=_sha(tmp_path / "decl.json"),
                               instrument_dir="kit")
        result = er.end_read(args)
        assert (result["verdict"], result["reason"]) == (
            "INCONCLUSIVE", "the declaration has no settings.lcm ignore and stateless session patterns"), decl


def test_condition_2_rows_in_a_declared_pattern_session_are_counted_apart_never_lost():
    by_design = er.session_matchers({"ignore_session_patterns": ["cron_*"], "stateless_session_patterns": []}, {})[0]
    end = {"host": [*START["host"], host(2, "cron_7", "scheduled"), host(3, "cron_7", "more"), host(4, "s1", "a")],
           "store": [*START["store"], stored(11, "s1", "a")]}
    check2 = er.coverage(START, end, identity, "/nonexistent", by_design)
    assert (check2["not_stored_by_design"]["sessions"], check2["not_stored_by_design"]["rows"]) == (1, 2)
    assert (check2["lost"], check2["unmatched"], check2["matched"], check2["window_host_rows"]) == (0, 0, 1, 1)
    assert er.verdict(PRESERVED, check2) == "PASS"
    only_designed = {"host": [*START["host"], host(2, "cron_7", "scheduled")], "store": [*START["store"]]}
    assert er.verdict(PRESERVED, er.coverage(START, only_designed, identity, "/x", by_design)) == "INCONCLUSIVE"


def test_condition_2_a_platform_key_matches_as_the_engine_matches():
    by_design = er.session_matchers({"ignore_session_patterns": [], "stateless_session_patterns": ["cron:*"]},
                                     {"s5": "cron"})[0]
    assert by_design("s5") and not by_design("s1")


def test_condition_3_a_declared_pattern_session_with_a_stored_row_is_inconclusive():
    by_design = er.session_matchers({"ignore_session_patterns": ["cron_*"], "stateless_session_patterns": []}, {})[0]
    end = {"host": [*START["host"], host(2, "cron_7", "x"), host(3, "s1", "a")],
           "store": [*START["store"], stored(11, "s1", "a"), stored(12, "cron_7", "x")]}
    check2 = er.coverage(START, end, identity, "/nonexistent", by_design)
    assert check2["declared_pattern_sessions_with_stored_rows"] == [er._sha12("cron_7")]
    assert er.verdict(PRESERVED, check2) == "INCONCLUSIVE"


def test_a_child_of_a_declared_pattern_session_is_never_by_design_and_never_lost():
    by_design, inherits = er.session_matchers({"ignore_session_patterns": ["cron_*"], "stateless_session_patterns": []},
                                              {}, {"child": "cron_7", "grandchild": "child", "loop": "loop"})
    assert inherits("child") and inherits("grandchild") and not inherits("s1") and not inherits("loop")
    assert not by_design("child")
    # A child with no stored row: unmatched with its hint (INCONCLUSIVE), never by design, never LOST, never PASS.
    end = {"host": [*START["host"], host(2, "child", "x"), host(3, "s1", "a")],
           "store": [*START["store"], stored(11, "s1", "a")]}
    check2 = er.coverage(START, end, identity, "/x", by_design, inherits)
    assert (check2["lost"], check2["unmatched"], check2["not_stored_by_design"]["rows"]) == (0, 1, 0)
    assert check2["unmatched_by_hint"]["session never stored; a parent session matches a declared pattern"] == 1
    assert er.verdict(PRESERVED, check2) == "INCONCLUSIVE"
    # A child the engine stored (a branch or delegate child) is matched like any session.
    end["store"].append(stored(12, "child", "x"))
    check2 = er.coverage(START, end, identity, "/x", by_design, inherits)
    assert (check2["matched"], check2["unmatched"], er.verdict(PRESERVED, check2)) == (2, 0, "PASS")


def test_a_stateless_pattern_alone_counts():
    assert er.session_matchers({"ignore_session_patterns": [], "stateless_session_patterns": ["tmp_**"]}, {})[0]("tmp_a:b")


def test_the_host_db_maps_are_read_only_and_tolerate_missing_columns(tmp_path):
    con = sqlite3.connect(tmp_path / "host-0.db")
    con.execute("CREATE TABLE sessions (id TEXT)")
    con.execute("INSERT INTO sessions VALUES ('s1')")
    con.commit()
    con.close()
    assert er._sessions(tmp_path) == ({}, {})
    con = sqlite3.connect(tmp_path / "host-0.db")
    con.execute("ALTER TABLE sessions ADD COLUMN source TEXT")
    con.execute("ALTER TABLE sessions ADD COLUMN parent_session_id TEXT")
    con.execute("INSERT INTO sessions VALUES ('s2', 'cron', 's1')")
    con.commit()
    con.close()
    before = sorted(p.name for p in tmp_path.iterdir())
    assert er._sessions(tmp_path) == ({"s1": None, "s2": "cron"}, {"s2": "s1"})
    assert sorted(p.name for p in tmp_path.iterdir()) == before  # No sidecar written beside the input.


def test_the_read_passes_the_declared_patterns_to_coverage(monkeypatch, tmp_path):
    seen = {}

    class Host:
        def __init__(self, config, scratch):
            pass

        def pins(self):
            return {"verdict": "PASS"}

        def rules(self):
            return {"identity": identity}

        def rows(self, folder):
            return START if folder == "start-dir" else {"host": [*START["host"], host(2, "cron_1", "x"),
                                                                 host(3, "branch_1", "y")],
                                                        "store": [*START["store"]]}

    real = er.coverage

    def coverage(*a, **k):
        seen["result"] = real(*a, **k)
        return seen["result"]
    monkeypatch.setattr(er, "_capture", lambda *a: ({"verdict": "PASS"}, None))
    monkeypatch.setattr(er.runner, "LocalHost", Host)
    monkeypatch.setattr(er, "_sessions", lambda folder: ({}, {"branch_1": "cron_1"}))
    monkeypatch.setattr(er, "coverage", coverage)
    lcm = {"ignore_session_patterns": ["cron_*"], "stateless_session_patterns": []}
    (tmp_path / "decl.json").write_text(json.dumps({"settings": {"lcm": lcm}}))
    args = SimpleNamespace(start="start-dir", start_pin="p", start_pin_sha256="a" * 64, end="end-dir", end_pin="p",
                           end_pin_sha256="b" * 64, scratch=str(tmp_path), config=str(tmp_path / "decl.json"), config_sha256=_sha(tmp_path / "decl.json"),
                           instrument_dir="kit")
    result = er.end_read(args)
    assert (seen["result"]["not_stored_by_design"]["sessions"], seen["result"]["not_stored_by_design"]["rows"]) == (1, 1) and seen["result"]["lost"] == 0
    assert seen["result"]["unmatched_by_hint"]["session never stored; a parent session matches a declared pattern"] == 1
    assert result["verdict"] == "INCONCLUSIVE"  # Only by-design rows: the window of other rows is empty.


def test_a_declared_pattern_session_with_stored_rows_outside_the_window_is_inconclusive():
    by_design = er.session_matchers({"ignore_session_patterns": ["cron_*"], "stateless_session_patterns": []}, {})[0]
    start = {"host": START["host"], "store": [*START["store"], stored(5, "cron_7", "old")]}
    end = {"host": [*start["host"], host(2, "s1", "a")], "store": [*start["store"], stored(11, "s1", "a")]}
    check2 = er.coverage(start, end, identity, "/x", by_design)
    assert check2["declared_pattern_sessions_with_stored_rows"] == [er._sha12("cron_7")]
    assert er.verdict(PRESERVED, check2) == "INCONCLUSIVE"


def test_the_declaration_must_hash_to_the_sha256_sent(monkeypatch, tmp_path):
    monkeypatch.setattr(er, "_capture", lambda *a: ({"verdict": "PASS"}, None))
    (tmp_path / "decl.json").write_text(json.dumps({"settings": {"lcm": {}}}))
    args = SimpleNamespace(start="a", start_pin="p", start_pin_sha256="a" * 64, end="b", end_pin="p",
                           end_pin_sha256="b" * 64, scratch=str(tmp_path), config=str(tmp_path / "decl.json"),
                           config_sha256="0" * 64, instrument_dir="kit")
    assert er.end_read(args)["reason"] == "the declaration does not hash to the sha256 sent"


def test_the_excluded_sessions_are_listed_by_hash():
    by_design = er.session_matchers({"ignore_session_patterns": ["cron_*"], "stateless_session_patterns": []}, {})[0]
    end = {"host": [*START["host"], host(2, "cron_7", "x"), host(3, "s1", "a")],
           "store": [*START["store"], stored(11, "s1", "a")]}
    check2 = er.coverage(START, end, identity, "/x", by_design)
    assert check2["not_stored_by_design"]["session_sha256_12"] == [er._sha12("cron_7")]
