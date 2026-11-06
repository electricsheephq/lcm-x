import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.golden_outcome_gate import OUTCOME_HEADS, compare, golden, outcome_sha, outcomes


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/golden_outcome_gate.py"


@pytest.fixture
def repo(tmp_path):
    def git(*args):
        subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)

    fixtures = tmp_path / "tests" / "fixtures"
    fixtures.mkdir(parents=True)
    path = fixtures / "ingest_trace.json"
    path.write_text(json.dumps({"case": ["sql old", "attr old", "state-after old"]}))
    (fixtures / "golden_outcome_changes.json").write_text("{}")
    git("init", "--quiet")
    git("add", ".")
    git("-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-qm", "base")
    return tmp_path, path


def run(repo, *args, family="ingest"):
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--base-ref", "HEAD", "--families", family, *args],
        cwd=repo, capture_output=True, text=True,
    )


def test_sql_attributes_and_helpers_do_not_change_outcomes(repo):
    root, path = repo
    path.write_text(json.dumps({"case": ["attr new", "sql new", "store result new", "state-after old"]}))
    assert run(root).returncode == 0


def test_changed_outcome_requires_exact_digest(repo):
    root, path = repo
    events = ["state-after new", "return-0 []"]
    path.write_text(json.dumps({"case": ["sql new", *events]}))
    result = run(root)
    digest = hashlib.sha256("\n".join(events).encode()).hexdigest()
    assert result.returncode == 1
    assert "base: state-after old\n  new:  state-after new" in result.stdout
    assert f'"outcome_sha": "{digest}"' in result.stdout
    allowlist = path.with_name("golden_outcome_changes.json")
    allowlist.write_text(json.dumps({"ingest": {"case": {"outcome_sha": digest, "reason": "Changed behaviour"}}}))
    assert run(root).returncode == 0
    path.write_text(json.dumps({"case": ["state-after later"]}))
    result = run(root)
    assert result.returncode == 1
    assert "stale allowlist entry" in result.stdout


@pytest.mark.parametrize("events", [[], ["state-after value"]])
def test_added_and_removed_scenarios(repo, events):
    root, path = repo
    path.write_text(json.dumps({"added": events}))
    result = run(root)
    assert result.returncode == 1
    assert "ingest/added: added" in result.stdout
    assert "ingest/case: removed" in result.stdout
    entries = {name: {"outcome_sha": outcome_sha(rows), "reason": "Scenario inventory change"}
               for name, rows in (("added", events), ("case", []))}
    path.with_name("golden_outcome_changes.json").write_text(json.dumps({"ingest": entries}))
    assert run(root).returncode == 0


def test_stale_entry_without_difference_is_advisory(repo):
    root, path = repo
    path.with_name("golden_outcome_changes.json").write_text(json.dumps({
        "ingest": {"case": {"outcome_sha": "0" * 64, "reason": "Earlier behaviour change"}},
    }))
    result = run(root)
    assert result.returncode == 0
    assert "stale allowlist entry" in result.stdout


@pytest.mark.parametrize("family", OUTCOME_HEADS)
def test_every_outcome_head_is_gated_and_noise_is_ignored(family):
    for head in OUTCOME_HEADS[family]:
        before, after = ([head, "old"], [head, "new"]) if family == "summary_chain" else (head + "old", head + "new")
        assert compare(family, {"case": [before]}, {"case": [after]}, {}) == 1
    if family in NEW_FAMILY_NOISE:
        assert outcomes(family, NEW_FAMILY_NOISE[family]) == []
        return
    assert outcomes(family, ["sql changed", "attr changed", "attrs changed", "log changed",
                             "before changed", "after changed", "helper return changed",
                             "attr _store_complete_cut=changed", "attr _compress_run=changed",
                             "after _compress_run changed", "before _store changed"]) == []


# Written from the harnesses, not from OUTCOME_HEADS: events that snapshot stored rows or final state.
STORED_STATE_EVENTS = {
    "ingest": ["state-after rows=1", "cold-state rows=1"],
    "compress": ["attr _store=sha12/len=0/1", "attr _dag=[]", "attr _lifecycle=sha12/len=0/1",
                 "attr _ingest_cursor=0"],
    "reconcile": ["attr _store=sha12/len=0/1", "attr _dag=[]", "attr _lifecycle=sha12/len=0/1",
                  "attr _ingest_cursor=0"],
    "assembly": ["after _store sha256=0 chars=1", "after _dag sha256=0 chars=1"],
    "summary_chain": [["state", {"model": "primary"}, None, True]],
    "backfill": ["after sha12/len=abc/123", "reopened state sha12/len=abc/123",
                 "accepted checkpoint sha12/len=abc/123", "discovery state sha12/len=abc/123"],
}


# Independent examples from test_summary_chain_trace.py and backfill_trace_harness.py.
NEW_FAMILY_NOISE = {
    "summary_chain": [[head, "changed"] for head in (
        "accepts", "admit", "allows", "hung_route", "in_config_episode", "log", "note_route",
        "provenance", "provider", "record_call", "record_config_error", "record_failure",
        "record_hang", "record_rejection", "record_success", "spend", "stop", "return_helper",
    )],
    "backfill": ["sql changed", "log changed", "store enter changed", "store exit changed",
                 "store raises changed", "connection changed", "provider changed", "files changed",
                 "discovery checkpoint changed", "before changed", "observer changed", "boundary changed",
                 "callback changed", "document changed", "reopen changed", "race changed",
                 "commit raises changed", "teardown raises changed"],
}


@pytest.mark.parametrize("family,event", [(family, event) for family, events in STORED_STATE_EVENTS.items()
                                          for event in events])
def test_stored_state_changes_fail_unlisted(family, event):
    changed = [*event, "changed"] if isinstance(event, list) else event + " changed"
    assert compare(family, {"case": [event]}, {"case": [changed]}, {}) == 1


@pytest.mark.parametrize("family,event", [
    ("summary_chain", ["return", "short"]), ("summary_chain", ["raise", "SweepBudgetExhausted", "budget"]),
    ("backfill", "return type [\"str\",null]"), ("backfill", "report fields changed"),
    ("backfill", "rendered changed"), ("backfill", "runner raises changed"),
    *[(family, event) for family in ("summary_chain", "backfill") for event in STORED_STATE_EVENTS[family]],
])
def test_new_family_outcome_requires_printed_digest(repo, family, event):
    root, old_path = repo
    path = old_path.with_name(f"{family}_trace.json")
    before = [*event, "old"] if isinstance(event, list) else event + " old"
    path.write_text(json.dumps({"case": [before]}))
    subprocess.run(["git", "add", str(path)], cwd=root, check=True, capture_output=True)
    subprocess.run(["git", "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                    "commit", "-qm", "Add family base"], cwd=root, check=True, capture_output=True)
    path.write_text(json.dumps({"case": [event]}))
    result = run(root, family=family)
    assert result.returncode == 1
    digest = next(line for line in result.stdout.splitlines() if '"outcome_sha":' in line)
    entry = json.loads("{" + digest + ', "reason": "Accepted outcome change"}')
    canonical = json.dumps(event, sort_keys=True, separators=(",", ":")) if isinstance(event, list) else event
    assert entry["outcome_sha"] == hashlib.sha256(canonical.encode()).hexdigest()
    path.with_name("golden_outcome_changes.json").write_text(json.dumps({family: {"case": entry}}))
    assert run(root, family=family).returncode == 0
    path.write_text(json.dumps({"case": [event, event]}))
    assert run(root, family=family).returncode == 1


@pytest.mark.parametrize("family", ["summary_chain", "backfill"])
def test_new_family_noise_changes_pass(family):
    current = golden((ROOT / f"tests/fixtures/{family}_trace.json").read_text(encoding="utf-8"))
    changed = {name: [*NEW_FAMILY_NOISE[family], *reversed([
        event for event in events if not outcomes(family, [event])
    ]), *[event for event in events if outcomes(family, [event])]] for name, events in current.items()}
    assert compare(family, current, changed, {}) == 0


def test_list_events_are_canonical_and_use_exact_head():
    first = golden('{"case": [["state", {"z": 1, "a": 2}, null, true]]}')
    second = golden('{"case": [["state", {"a": 2, "z": 1}, null, true]]}')
    assert compare("summary_chain", first, second, {}) == 0
    assert outcomes("summary_chain", first["case"]) == ['["state",{"a":2,"z":1},null,true]']
    assert outcomes("summary_chain", [["state_helper", 1]]) == []


@pytest.mark.parametrize("target,text", [("ingest_trace.json", "[]"),
                                       ("ingest_trace.json", '{"case": [42]}'),
                                       ("ingest_trace.json", '{"case": [[]]}'),
                                       ("ingest_trace.json", '{"case": [[42]]}'),
                                       ("golden_outcome_changes.json", "{"),
                                       ("golden_outcome_changes.json", '{"ingest": {"case": {}}}')])
def test_parse_errors_exit_two(repo, target, text):
    root, path = repo
    path.with_name(target).write_text(text)
    assert run(root).returncode == 2


def test_usage_errors_exit_two(repo):
    root, _ = repo
    assert run(root, "--families", "unknown").returncode == 2
    assert run(root, "--base-ref", "missing-ref").returncode == 2


@pytest.mark.parametrize("family", OUTCOME_HEADS)
def test_current_golden_compared_with_itself(family):
    current = golden((ROOT / f"tests/fixtures/{family}_trace.json").read_text(encoding="utf-8"))
    assert compare(family, current, current, {}) == 0
    assert all(outcomes(family, events) for events in current.values())
