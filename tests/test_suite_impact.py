import importlib.util
from itertools import takewhile
import json
from pathlib import Path
import subprocess
import sys

import pytest

# Load the script from its file. host-suite.yml puts a Hermes source tree on PYTHONPATH, and Hermes ships
# scripts/release.py, a module that wins over this repo's scripts/release/ folder in `import scripts.release...`.
_SPEC = importlib.util.spec_from_file_location(
    "lcmx_suite_impact", Path(__file__).resolve().parents[1] / "scripts" / "release" / "suite_impact.py"
)
_SUITE_IMPACT = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _SUITE_IMPACT
_SPEC.loader.exec_module(_SUITE_IMPACT)
MAP, ROOT, MapError, load_map, main = (
    _SUITE_IMPACT.MAP, _SUITE_IMPACT.ROOT, _SUITE_IMPACT.MapError, _SUITE_IMPACT.load_map, _SUITE_IMPACT.main
)
compile_glob, decide, literal_globs = _SUITE_IMPACT.compile_glob, _SUITE_IMPACT.decide, _SUITE_IMPACT.literal_globs

REAL = load_map(MAP)
MAPPED = {s["id"] for s in REAL["suites"] if s["kind"] == "mapped"}
MANUAL = {s["id"] for s in REAL["suites"] if s["kind"] == "manual"}
CLI = [sys.executable, str(ROOT / "scripts/release/suite_impact.py")]


def selected(decision, status):
    return {suite_id for suite_id, verdict, _ in decision.verdicts if verdict == status}


@pytest.mark.parametrize("paths", [
    ["brand_new_dir/x.txt"], ["newmod.py"], ["engine.py", "engine_core.py"], ["claude_code_new.py"],
])
def test_unmapped_and_rename(paths):
    result = decide(REAL, paths)
    assert result.unmapped == paths[-1:]
    assert (selected(result, "RUN"), selected(result, "MANUAL")) == (MAPPED, MANUAL)


def test_fail_closed_path():
    result = decide(REAL, ["sanitize.py"])
    assert result.fail_closed == ["sanitize.py"] and not result.unmapped
    assert selected(result, "RUN") == MAPPED


def test_notes_only():
    result = decide(REAL, [".github/release-notes/v9.9.9.md"])
    # These are the notes checks.
    notes_checks = {"c5", "release-yml", "ci-tests", "host-suite"}
    assert selected(result, "RUN") == notes_checks
    assert selected(result, "MANUAL") == MANUAL
    assert {sid: reason for sid, verdict, reason in result.verdicts if verdict == "SKIP"} == {
        s["id"]: f"suite {s['name']} skipped: nothing it measures changed"
        for s in REAL["suites"] if s["id"] in MAPPED - notes_checks}


def test_tracked_paths():
    paths = subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True).splitlines()
    result = decide(REAL, paths)
    assert not result.unmapped and selected(result, "RUN") == MAPPED and not selected(result, "SKIP")
    globs = literal_globs(REAL) | {glob for values in REAL["path_sets"].values() for glob in values}
    assert not {glob for glob in globs if not any(map(compile_glob(glob).fullmatch, paths))}


def table(doc, name):
    body = doc.split(f"<!-- {name} TABLE -->\n", 1)[1]
    return [line[2:-2].split(" | ") for line in takewhile(lambda line: line.startswith("|"), body.splitlines()[2:])]


def items(cell):
    return [] if cell == "—" else [item.strip("`") for item in cell.split(", ")]


@pytest.mark.parametrize("drift", [None, "cells", "duplicate-list-row"])
def test_doc_matches_json(drift):
    doc = MAP.with_suffix(".md").read_text(encoding="utf-8")
    inputs, lists = table(doc, "INPUTS"), table(doc, "LISTS")
    mechanisms = table(doc, "MECHANISMS")
    suites = table(doc, "SUITES")
    if drift == "cells":
        for rows, column in ((inputs, 1), (lists, 1), (mechanisms, 3), (suites, 6)):
            rows[0][column] = "Changed"
    elif drift == "duplicate-list-row":
        lists.append(lists[0])
    assert ({key.strip("`"): value for key, value in inputs} == REAL["inputs"]) is (drift != "cells")
    expected_lists = [("@" + key, value) for key, value in REAL["path_sets"].items()]
    expected_lists.extend((key, REAL[key]) for key in ("fail_closed_paths", "ignored_paths"))
    assert ([(key.strip("`"), items(value)) for key, value in lists] == expected_lists) is (drift is None)
    actual = [(key.strip("`"), *map(items, cells[2:])) for key, *cells in mechanisms]
    expected = [(m["id"], m["paths"], m["symbols"], [s["id"] for s in REAL["suites"] if m["id"] in s["mechanisms"]])
                for m in REAL["mechanisms"]]
    assert (actual == expected) is (drift != "cells")
    actual = [(key.strip("`"), kind, *map(items, cells[:3]), cells[3]) for key, _, kind, *cells in suites]
    expected = [tuple(suite[key] for key in ("id", "kind", "mechanisms", "paths", "inputs", "when"))
                for suite in REAL["suites"]]
    assert (actual == expected) is (drift != "cells")


def test_glob_rules():
    for glob, path, matched in [("**/*.md", "README.md", True), ("**/*.md", "a/b/c.md", True),
                                ("*.py", "tests/x.py", False), ("bench/**", "bench/a/b", True), ("?", "/", False)]:
        assert bool(compile_glob(glob).fullmatch(path)) == matched


def assert_bad_map(path, m):
    path.write_text(json.dumps(m))
    with pytest.raises(MapError):
        load_map(path)
    result = subprocess.run([*CLI, "HEAD", "HEAD", "--map", str(path)], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 2 and result.stderr.startswith("BAD MAP: ") and not result.stdout


@pytest.mark.parametrize("changes", [
    None, "name", {"mechanisms": ["M99"]}, {"paths": ["@unknown"]}, {"paths": []}, {"paths": "known.py"},
    {"kind": "typo"}, {"inputs": ["unknown"]}, {"mechanisms": "M01"}, {"inputs": [1]}, {"paths": [1]},
])
def test_bad_maps(tmp_path, changes):
    suite = dict(id="s", name="Suite", kind="mapped", when="By hand", note="", paths=["known.py"], inputs=[], mechanisms=[])
    if changes == "name":
        del suite["name"]
    elif changes is not None:
        suite.update(changes)
    m = dict(version=1, glob_rules="", inputs={}, path_sets={}, fail_closed_paths=[], ignored_paths=[],
             mechanisms=[], suites=[suite, suite] if changes is None else [suite])
    assert_bad_map(tmp_path / "map.json", m)


@pytest.mark.parametrize("changes", [
    {"version": 99}, {"suites": []}, {"inputs": []}, {"mechanisms": [{"id": "m", "paths": []}]},
    {"path_sets": {"empty": []}, "suites": [{**REAL["suites"][0], "mechanisms": [], "paths": ["@empty"], "inputs": []}]},
])
def test_bad_map_structure(tmp_path, changes):
    assert_bad_map(tmp_path / "map.json", {**REAL, **changes})


def test_bad_input():
    with pytest.raises(MapError, match="unknown input"):
        decide(REAL, [], ["unknown"])


def test_cli(tmp_path):
    command = [*CLI, "HEAD", "HEAD"]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    expected = f"counts: run=0 skip={len(MAPPED)} manual={len(MANUAL)} unmapped_paths=0"
    assert result.returncode == 0 and result.stdout.splitlines()[-1] == expected
    bad = tmp_path / "bad.json"
    bad.write_text("{")
    result = subprocess.run([*command, "--map", str(bad)], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 2 and "BAD MAP" in result.stderr


@pytest.mark.parametrize("revisions", [("", "HEAD"), ("HEAD", ""), ("  ", "HEAD"), ("HEAD", " \t")])
def test_cli_rejects_empty_revision(monkeypatch, capsys, revisions):
    monkeypatch.setattr(sys, "argv", [CLI[1], *revisions])
    monkeypatch.setattr(subprocess, "check_output", lambda *a, **kw: pytest.fail("git ran"))
    assert main() == 1
    output = capsys.readouterr()
    assert not output.out and "base and head revisions must not be empty" in output.err


@pytest.mark.parametrize("line_break", ["\n", "\r"])
def test_cli_refuses_path_with_line_break(monkeypatch, capsys, line_break):
    path = f"file.py{line_break}SKIP forged"
    monkeypatch.setattr(sys, "argv", [CLI[1], "HEAD", "HEAD"])
    monkeypatch.setattr(subprocess, "check_output", lambda *a, **kw: f"engine.py\0{path}\0".encode())
    assert main() == 1
    output = capsys.readouterr()
    assert not output.out
    assert output.err == f"REFUSED {path!r}: path contains a line break; no run/skip list printed\n"


RELIABILITY_FILES = ["README.md", "ci.py", "test_ci_compare.py"] + [
    f"fixtures/compare/{stem}.jsonl" for stem in ("44230043164", "44451000342", "44667323840", "respin-base", "respin-final")
]


@pytest.mark.parametrize("paths, count, run, skip", [
    (["survival_codec.py", "survival_fit.py", "tests/invariants/test_issue_1221_objective_duplicate_after_2914.py"],
     26, {"c1a", "respin", "nightly-matrix", "real-host", "m3", "d7", "invariants"}, {"c4", "c2", "busy-bar"}),
    (["changelog.d/3006-post-refusal-replay.md", "compaction.py", "engine.py",
      "tests/invariants/test_issue_3006_post_refusal.py", "tests/test_refused_compaction_rollback.py"], 44, {"ci-bench"}, set()),
    (["changelog.d/1221-objective-replay-after-compaction.md", "changelog.d/2863-post-tool-reanchor.md",
      "compaction.py", "engine.py", "reconcile.py", "tests/invariants/test_issue_1221_objective_duplicate_after_2914.py",
      "tests/invariants/test_issue_2863_post_tool_reanchor.py", "tests/test_issue_2863_host_contract.py",
      "tests/test_issue_2863_landing.py"], 45, {"busy-bar"}, set()),
    ([".github/release-notes/v0.30.1-rc4.md"], 4, {"c5", "release-yml", "ci-tests", "host-suite"}, set()),
    ([".github/workflows/reliability-nightly.yml", "changelog.d/reliability-nightly-compare.md",
      "tests/test_reliability_workflow_artifacts.py",
      *["bench/instruments/reliability/" + file for file in RELIABILITY_FILES]], 15, {"real-host", "invariants"}, set()),
])
def test_proof_paths(paths, count, run, skip):
    result = decide(REAL, paths)
    assert len(selected(result, "RUN")) == count
    assert run <= selected(result, "RUN") and skip <= selected(result, "SKIP")
