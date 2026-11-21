from itertools import takewhile
import json
import subprocess
import sys

import pytest

from scripts.release.suite_impact import MAP, ROOT, MapError, compile_glob, decide, literal_globs, load_map

REAL = load_map(MAP)
MAPPED = {s["id"] for s in REAL["suites"] if s["kind"] == "mapped"}
MANUAL = {s["id"] for s in REAL["suites"] if s["kind"] == "manual"}


def selected(decision, status):
    return {suite_id for suite_id, verdict, _ in decision.verdicts if verdict == status}


@pytest.mark.parametrize("paths", [["brand_new_dir/x.txt"], ["newmod.py"], ["engine.py", "engine_core.py"]])
def test_unmapped_and_rename(paths):
    result = decide(REAL, paths)
    assert result.unmapped == paths[-1:]
    assert (selected(result, "RUN"), selected(result, "MANUAL")) == (MAPPED, MANUAL)


def test_notes_only():
    result = decide(REAL, [".github/release-notes/v9.9.9.md"])
    assert selected(result, "RUN") == {"c5", "release-yml"}
    assert selected(result, "MANUAL") == MANUAL
    assert {sid: reason for sid, verdict, reason in result.verdicts if verdict == "SKIP"} == {
        s["id"]: f"suite {s['name']} skipped: nothing it measures changed"
        for s in REAL["suites"] if s["id"] in MAPPED - {"c5", "release-yml"}}


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


@pytest.mark.parametrize("drift", [False, True])
def test_doc_matches_json(drift):
    doc = MAP.with_suffix(".md").read_text(encoding="utf-8")
    assert {key.strip("`") for key, _ in table(doc, "INPUTS")} == REAL["inputs"].keys()
    assert {key.strip("`"): items(value) for key, value in table(doc, "LISTS")} == {
        **{"@" + key: value for key, value in REAL["path_sets"].items()},
        "fail_closed_paths": REAL["fail_closed_paths"], "ignored_paths": REAL["ignored_paths"]}
    mechanisms = table(doc, "MECHANISMS")
    mechanisms[0][3] = "—" if drift else mechanisms[0][3]
    actual = [(key.strip("`"), *map(items, cells[2:])) for key, *cells in mechanisms]
    expected = [(m["id"], m["paths"], m["symbols"], [s["id"] for s in REAL["suites"] if m["id"] in s["mechanisms"]])
                for m in REAL["mechanisms"]]
    assert (actual == expected) is not drift
    assert [(key.strip("`"), kind, *map(items, cells[:3])) for key, _, kind, *cells in table(doc, "SUITES")] == [
        tuple(suite[key] for key in ("id", "kind", "mechanisms", "paths", "inputs")) for suite in REAL["suites"]
    ]


def test_glob_rules():
    for glob, path, matched in [("**/*.md", "README.md", True), ("**/*.md", "a/b/c.md", True),
                                ("*.py", "tests/x.py", False), ("bench/**", "bench/a/b", True), ("?", "/", False)]:
        assert bool(compile_glob(glob).fullmatch(path)) == matched


@pytest.mark.parametrize("changes", [None, {"mechanisms": ["M99"]}, {"paths": ["@unknown"]},
                                      {"paths": []}, {"paths": "known.py"}, {"kind": "typo"}, {"inputs": ["unknown"]}])
def test_bad_maps(tmp_path, changes):
    suite = dict(id="s", name="Suite", kind="mapped", note="", paths=["known.py"], inputs=[], mechanisms=[])
    m = dict(version=1, glob_rules="", inputs={}, path_sets={}, fail_closed_paths=[], ignored_paths=[],
             mechanisms=[], suites=[suite, suite] if changes is None else [{**suite, **changes}])
    (tmp_path / "map.json").write_text(json.dumps(m))
    with pytest.raises(MapError):
        load_map(tmp_path / "map.json")


def test_bad_input():
    with pytest.raises(MapError, match="unknown input"):
        decide(REAL, [], ["unknown"])


def test_cli(tmp_path):
    command = [sys.executable, str(ROOT / "scripts/release/suite_impact.py"), "HEAD", "HEAD"]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    expected = f"counts: run=0 skip={len(MAPPED)} manual={len(MANUAL)} unmapped_paths=0"
    assert result.returncode == 0 and result.stdout.splitlines()[-1] == expected
    bad = tmp_path / "bad.json"
    bad.write_text("{")
    result = subprocess.run([*command, "--map", str(bad)], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 2 and "BAD MAP" in result.stderr


RELIABILITY_FILES = ["README.md", "ci.py", "test_ci_compare.py"] + [
    f"fixtures/compare/{stem}.jsonl" for stem in ("44230043164", "44451000342", "44667323840", "respin-base", "respin-final")
]


@pytest.mark.parametrize("paths, count, run, skip", [
    (["survival_codec.py", "survival_fit.py", "tests/invariants/test_issue_1221_objective_duplicate_after_2914.py"],
     24, {"c1a", "respin", "nightly-matrix", "real-host", "m3", "d7"}, {"c4", "c2", "busy-bar"}),
    (["changelog.d/3006-post-refusal-replay.md", "compaction.py", "engine.py",
      "tests/invariants/test_issue_3006_post_refusal.py", "tests/test_refused_compaction_rollback.py"], 43, set(), set()),
    (["changelog.d/1221-objective-replay-after-compaction.md", "changelog.d/2863-post-tool-reanchor.md",
      "compaction.py", "engine.py", "reconcile.py", "tests/invariants/test_issue_1221_objective_duplicate_after_2914.py",
      "tests/invariants/test_issue_2863_post_tool_reanchor.py", "tests/test_issue_2863_host_contract.py",
      "tests/test_issue_2863_landing.py"], 43, {"busy-bar"}, set()),
    ([".github/release-notes/v0.30.1-rc4.md"], 2, {"c5", "release-yml"}, set()),
    ([".github/workflows/reliability-nightly.yml", "changelog.d/reliability-nightly-compare.md",
      "tests/test_reliability_workflow_artifacts.py",
      *["bench/instruments/reliability/" + file for file in RELIABILITY_FILES]], 11, set(), set()),
])
def test_proof_paths(paths, count, run, skip):
    result = decide(REAL, paths)
    assert len(selected(result, "RUN")) == count
    assert run <= selected(result, "RUN") and skip <= selected(result, "SKIP")
