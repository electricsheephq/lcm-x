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


def run(repo, *args):
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--base-ref", "HEAD", "--families", "ingest", *args],
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
        assert compare(family, {"case": [head + "old"]}, {"case": [head + "new"]}, {}) == 1
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
}


@pytest.mark.parametrize("family,event", [(family, event) for family, events in STORED_STATE_EVENTS.items()
                                          for event in events])
def test_stored_state_changes_fail_unlisted(family, event):
    assert compare(family, {"case": [event]}, {"case": [event + " changed"]}, {}) == 1


@pytest.mark.parametrize("target,text", [("ingest_trace.json", "[]"),
                                       ("ingest_trace.json", '{"case": [42]}'),
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
