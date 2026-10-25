"""Offline cell scoring uses an explicitly verified plugin checkout."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from bench.instruments.reliability.scorers import cli


@pytest.fixture
def plugin(tmp_path):
    tree = tmp_path / "plugin"
    tree.mkdir()
    subprocess.run(["git", "init", "-q", str(tree)], check=True)
    (tree / "tracked").write_text("clean\n")
    subprocess.run(["git", "-C", str(tree), "add", "tracked"], check=True)
    subprocess.run(["git", "-C", str(tree), "-c", "user.name=Test", "-c",
                    "user.email=test@example.invalid", "commit", "-qm", "fixture", "--allow-empty"], check=True)
    sha = subprocess.check_output(["git", "-C", str(tree), "rev-parse", "HEAD"], text=True).strip()
    return tree, sha


def cell_dir(tmp_path, sha):
    cell = {"id": "fixture", "faults": [], "plugin": {"tree": "/hosted/plugin", "sha": sha, "ref": "HEAD"}}
    path = tmp_path / "cell.json"
    path.write_text(json.dumps(cell))
    return cell, path


def test_plugin_tree_override_reaches_bars_only_in_memory(tmp_path, plugin, monkeypatch, capsys):
    tree, sha = plugin
    cell, path = cell_dir(tmp_path, sha)
    seen = []
    monkeypatch.setattr(cli.bars, "score", lambda c, d: seen.append((c, d)) or {"verdict": "FAIL"})
    assert cli.main(["--cell-dir", str(tmp_path), "--plugin-tree", str(tree)]) == 0
    assert seen == [({**cell, "plugin": {**cell["plugin"], "tree": str(tree)}}, tmp_path)]
    assert json.loads(path.read_text()) == cell
    assert json.loads(capsys.readouterr().out) == {"verdict": "FAIL"}


@pytest.mark.parametrize("checkout", [True, False])
def test_plugin_tree_refuses_wrong_sha_or_non_checkout(tmp_path, plugin, monkeypatch, capsys, checkout):
    tree, _ = plugin
    _, path = cell_dir(tmp_path, "0" * 40)
    original = path.read_bytes()
    if not checkout:
        tree = tmp_path / "not-a-checkout"
        tree.mkdir()
    monkeypatch.setattr(cli.bars, "score", lambda *_: pytest.fail("unverified tree reached scorer"))
    with pytest.raises(SystemExit) as exc:
        cli.main(["--cell-dir", str(tmp_path), "--plugin-tree", str(tree)])
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "--plugin-tree" in err
    assert "HEAD" in err
    assert path.read_bytes() == original


def test_without_plugin_tree_preserves_cell_and_output(tmp_path, monkeypatch, capsys):
    cell, path = cell_dir(tmp_path, "unavailable-sha")
    seen = []
    report = {"verdict": "FAIL", "numbers": {"B1": {"mismatches": 15}}}
    monkeypatch.setattr(cli.bars, "score", lambda c, d: seen.append((c, d)) or report)
    assert cli.main(["--cell-dir", str(tmp_path)]) == 0
    assert seen == [(cell, tmp_path)]
    assert json.loads(path.read_text()) == cell
    assert capsys.readouterr().out == json.dumps(report, indent=1, default=str) + "\n"


@pytest.mark.parametrize("transport", [None, "acp-process"])
def test_r2_restores_uploaded_session_and_transport(tmp_path, monkeypatch, transport):
    cell, path = cell_dir(tmp_path, "unused")
    if transport:
        cell["transport"] = transport
    path.write_text(json.dumps(cell))
    (tmp_path / "verdict.json").write_text(json.dumps({"acp_session": "session-r2", "transport": "acp-process"}))
    seen = []
    monkeypatch.setattr(cli.bars, "score", lambda c, d: seen.append(c) or {"verdict": "PASS"})
    assert cli.main(["--cell-dir", str(tmp_path)]) == 0
    assert seen == [{**cell, "chat_root": "session-r2", "transport": "acp-process"}]
    assert json.loads(path.read_text()) == cell


@pytest.mark.parametrize("case,cell_fields,verdict,message", [
    ("missing-session", {"transport": "acp-process"}, {}, "acp_session"),
    ("stop-turn", {"stop_after_tool": True}, {}, "stop_after_tool"),
    ("f4-bar", {"id": "continuity/test"}, {"verdict": "FAIL", "failed_bars": {"F4": {}}}, "F4"),
    ("f4-reason", {"id": "continuity/test"}, {"verdict": "FAIL", "reason": "continuity F4: missing"}, "F4"),
    ("unattributed-fail", {"id": "continuity/test"}, {"verdict": "FAIL", "failed_bars": {}}, "no failed bars"),
    ("pre-run", None, {"verdict": "UNSUPPORTED"}, "no cell.json"),
])
def test_unreconstructable_cells_refused(tmp_path, monkeypatch, capsys, case, cell_fields, verdict, message):
    if cell_fields is not None:
        cell, path = cell_dir(tmp_path, "unused")
        path.write_text(json.dumps({**cell, **cell_fields}))
    (tmp_path / "verdict.json").write_text(json.dumps(verdict))
    monkeypatch.setattr(cli.bars, "score", lambda *_: pytest.fail("refused cell reached bars"))
    with pytest.raises(SystemExit) as exc:
        cli.main(["--cell-dir", str(tmp_path)])
    assert exc.value.code == 2
    assert message in capsys.readouterr().err


@pytest.mark.parametrize("kind", ["publication_failure", "crash_between_session_end_and_start"])
def test_never_fired_gate_uses_harness_reason(tmp_path, monkeypatch, capsys, kind):
    from bench.instruments.reliability.run_matrix import unfired_reason

    cell, path = cell_dir(tmp_path, "unused")
    cell["faults"] = [{"kind": kind}]
    path.write_text(json.dumps(cell))
    citations = {"rotation_start": "fixture rotation", "session_transition": "fixture transition"}
    (tmp_path / "phase-A.json").write_text(json.dumps({"citations": citations}))
    (tmp_path / "faults-fired.jsonl").write_text('{"kind":"other_fault"}\n')
    monkeypatch.setattr(cli.bars, "score", lambda *_: pytest.fail("never-fired cell reached bars"))
    assert cli.main(["--cell-dir", str(tmp_path)]) == 0
    assert json.loads(capsys.readouterr().out) == {
        "verdict": "UNSUPPORTED", "reason": unfired_reason(cell, {"other_fault"}, citations)}


@pytest.mark.parametrize("kind", ["subdir", "dirty", "GIT_DIR", "GIT_WORK_TREE"])
def test_plugin_tree_refuses_unsafe_checkout(tmp_path, plugin, monkeypatch, capsys, kind):
    tree, sha = plugin
    cell_dir(tmp_path, sha)
    if kind == "subdir":
        tree = tree / "subdir"
        tree.mkdir()
    elif kind == "dirty":
        (tree / "tracked").write_text("dirty\n")
    else:
        monkeypatch.setenv("GIT_DIR", str(tree / ".git"))
        if kind == "GIT_WORK_TREE":
            monkeypatch.setenv("GIT_WORK_TREE", str(tree))
        tree = tmp_path / "notgit"
        tree.mkdir()
    monkeypatch.setattr(cli.bars, "score", lambda *_: pytest.fail("unsafe checkout reached bars"))
    with pytest.raises(SystemExit) as exc:
        cli.main(["--cell-dir", str(tmp_path), "--plugin-tree", str(tree)])
    assert exc.value.code == 2
    assert "--plugin-tree" in capsys.readouterr().err


@pytest.mark.parametrize("mixed_prefixes", [False, True])
def test_cell_report_is_byte_identical_across_hash_seeds(tmp_path, mixed_prefixes):
    from tests.test_reliability_scorers import U, clean_events, clean_rows, held, make

    # Extend the existing single-licence fixture past the ten-record sample limit.
    tree = {"tree": str(Path.cwd())}
    duplicates = [("user", U.format(t, t)) for t in range(1, 13)]
    if mixed_prefixes:
        duplicates = [("user", "\n".join(f"[{p}{t:02d}] user turn" for p in "CDEFGH")) for t in (2, 1)] + duplicates
    make(tmp_path, rows=clean_rows(12) + duplicates, events=clean_events(12),
         host=held(12) + [("S0", role, text, 1) for role, text in duplicates], plugin=tree, bars=["B1", "B2"])
    directory = tmp_path / "cell"
    (directory / "cell.json").write_text(json.dumps({"plugin": tree, "bars": ["B1", "B2"]}))
    reports = [subprocess.check_output(
        [sys.executable, "-m", "bench.instruments.reliability.scorers.cli", "--cell-dir", str(directory)],
        env={**os.environ, "PYTHONHASHSEED": str(seed)}) for seed in (0, 1)]
    report = json.loads(reports[0])
    assert report["verdict"] == "PASS"
    assert report["numbers"]["B1"]["host_parity_licensed"]["rows"] == (24 if mixed_prefixes else 12)
    assert len(report["numbers"]["B1"]["host_parity_licensed"]["records"]) == 10
    if mixed_prefixes:
        examples = report["numbers"]["diagnostic"]["chronology"]["examples"]
        assert [e["tag"] for e in examples[:6]] == [f"{p}01" for p in "CDEFGH"]
    assert reports[0] == reports[1]
