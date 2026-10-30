"""CI evidence distinguishes executing a test from collecting or skipping it."""
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import xml.etree.ElementTree as ET

import pytest

from scripts.ci.check_tests_ran import changed_tests, check


@pytest.fixture
def evidence(tmp_path):
    allowlist = tmp_path / "allowlist.txt"
    allowlist.write_text("")

    def report(name="report.xml", cases=(("tests/test_a.py", "passed"),), empty=()):
        root = ET.Element("testsuites")
        suite = ET.SubElement(root, "testsuite")
        for path in empty:
            ET.SubElement(root, "testsuite", file=path, tests="0")
        for path, outcome in cases:
            node = ET.SubElement(suite, "testcase", file=path, name="test_example")
            if outcome != "passed":
                ET.SubElement(node, outcome)
        target = tmp_path / name
        ET.ElementTree(root).write(target)
        return target

    return tmp_path, allowlist, report


@pytest.mark.parametrize("outcome", ["passed", "failure", "error"])
def test_executed(evidence, outcome):
    root, allowlist, report = evidence
    errors, summary = check(root, [report(cases=(("tests/test_a.py", outcome),))],
                            ["tests/test_a.py"], allowlist)
    assert errors == []
    assert "tests/test_a.py | 1 | executed" in summary


def test_collected_zero(evidence):
    root, allowlist, report = evidence
    errors, _ = check(root, [report(empty=("tests/test_empty.py",))],
                      ["tests/test_empty.py"], allowlist)
    assert errors == ["tests/test_empty.py: collected with zero tests"]


def test_all_skipped(evidence):
    root, allowlist, report = evidence
    errors, summary = check(root, [report(cases=(("tests/test_a.py", "skipped"),))],
                            ["tests/test_a.py"], allowlist)
    assert "tests/test_a.py: all skipped" in errors
    assert any("zero executed tests" in error for error in errors)
    assert "tests/test_a.py | 0 | all skipped" in summary


def test_one_matrix_leg_executed_is_enough(evidence):
    root, allowlist, report = evidence
    skipped = report("skipped.xml", (("tests/test_a.py", "skipped"),
                                      ("tests/test_other.py", "passed")))
    errors, _ = check(root, [skipped, report()], ["tests/test_a.py"], allowlist)
    assert errors == []


def test_not_collected(evidence):
    root, allowlist, report = evidence
    errors, _ = check(root, [report()], ["bench/tests/test_b.py"], allowlist)
    assert errors == ["bench/tests/test_b.py: not collected by any CI job"]


def test_allowlisted_with_reason(evidence, capsys):
    root, allowlist, report = evidence
    allowlist.write_text("# resources\n\nbench/test_b.py  # requires a downloaded model\n")
    errors, summary = check(root, [report()], ["bench/test_b.py"], allowlist)
    assert errors == []
    assert "allowlisted: requires a downloaded model" in capsys.readouterr().out
    assert "Allowlisted: bench/test_b.py" in summary


@pytest.mark.parametrize("entry", ["bench/test_b.py", "bench/test_b.py #", "bench/test_b.py #  "])
def test_allowlist_empty_reason_errors(evidence, entry):
    root, allowlist, report = evidence
    allowlist.write_text(entry)
    errors, _ = check(root, [report()], [], allowlist)
    assert any("reason required" in error for error in errors)


@pytest.mark.parametrize("shape", ["missing", "malformed", "wrong-root", "no-file"])
def test_missing_or_malformed_junit_fails_closed(evidence, shape):
    root, allowlist, report = evidence
    target = root / "bad.xml"
    if shape == "malformed":
        target.write_text("<testsuite>")
    elif shape == "wrong-root":
        target.write_text("<unrelated />")
    elif shape == "no-file":
        target.write_text('<testsuite><testcase name="x" /></testsuite>')
    errors, _ = check(root, [report(), target], [], allowlist)
    assert any("missing or malformed JUnit" in error for error in errors)


def test_hypothesis_vendor_testcase_does_not_match_repo_file(evidence):
    root, allowlist, report = evidence
    errors, _ = check(root, [report(cases=(("../site-packages/hypothesis/stateful.py", "passed"),))],
                      ["tests/test_a.py"], allowlist)
    assert errors == ["tests/test_a.py: not collected by any CI job"]


@pytest.mark.parametrize("flags", [[], ["-n", "2"], ["-n", "2", "--import-mode=importlib"]],
                         ids=["serial", "xdist", "importlib"])
def test_generated_state_machine_credits_collecting_module(evidence, flags):
    root, allowlist, _ = evidence
    module = root / "generated/test_stateful.py"
    module.parent.mkdir()
    module.write_text(
        "from hypothesis import settings\n"
        "from hypothesis.stateful import RuleBasedStateMachine, rule\n"
        "class Machine(RuleBasedStateMachine):\n"
        "    @rule()\n"
        "    def step(self): assert True\n"
        "TestMachine = Machine.TestCase\n"
        "TestMachine.settings = settings(max_examples=2, stateful_step_count=2, database=None)\n"
    )
    (root / "pytest.ini").write_text("[pytest]\ntmp_path_retention_policy = failed\n")
    report = root / "generated.xml"
    result = subprocess.run(
        [sys.executable, "-m", "pytest", str(module), "-q", *flags,
         "-p", "no:cacheprovider", "-o", "junit_family=xunit1", f"--junitxml={report}",
         f"--basetemp={root / 'pytest-tmp'}"],
        cwd=root, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    case = next(ET.parse(report).iter("testcase"))
    assert case.get("classname") == "generated.test_stateful.TestMachine"
    assert "hypothesis/stateful.py" in case.get("file")
    errors, summary = check(root, [report], ["generated/test_stateful.py"], allowlist)
    assert errors == []
    assert "generated/test_stateful.py | 1 | executed" in summary


@pytest.mark.parametrize("file", [None, "../../../../../opt/hostedtoolcache/Python/3.12.15/"
                                "x64/lib/python3.12/site-packages/hypothesis/stateful.py"])
@pytest.mark.parametrize("skipped", [False, True])
def test_real_ci_state_machine_shape(evidence, file, skipped):
    root, allowlist, report = evidence
    path = "tests/invariants/test_exactly_once_machine.py"
    module = root / path
    module.parent.mkdir(parents=True)
    module.write_text("# collected repository module\n")
    target = report()
    tree = ET.parse(target)
    case = tree.find(".//testcase")
    case.attrib = {"classname": "tests.invariants.test_exactly_once_machine.TestExactlyOnce",
                   "name": "runTest", "line": "483", "time": "13.065"}
    if file is not None:
        case.set("file", file)
    if skipped:
        ET.SubElement(case, "skipped")
    tree.write(target)
    errors, summary = check(root, [target], [path], allowlist)
    if skipped:
        assert f"{path}: all skipped" in errors
    else:
        assert errors == []
        assert f"{path} | 1 | executed" in summary


def test_external_classname_does_not_credit_unrelated_repo_module(evidence):
    root, allowlist, report = evidence
    module = root / "tests/test_a.py"
    module.parent.mkdir()
    module.write_text("# unrelated\n")
    target = report(cases=(("../site-packages/hypothesis/stateful.py", "passed"),))
    tree = ET.parse(target)
    tree.find(".//testcase").set("classname", "unrelated.test_a.TestMachine")
    tree.write(target)
    errors, _ = check(root, [target], ["tests/test_a.py"], allowlist)
    assert errors == ["tests/test_a.py: not collected by any CI job"]


@pytest.mark.parametrize("xml", [
    '<testsuites><testcase file="tests/test_a.py"/></testsuites>',
    '<testsuites><wrapper><testcase file="tests/test_a.py"/></wrapper></testsuites>',
    '<testsuite><wrapper><testcase file="tests/test_a.py"/></wrapper></testsuite>',
    '<testsuite><testcase file="tests/test_a.py"><testcase file="tests/test_a.py"/></testcase></testsuite>',
])
def test_testcase_outside_direct_testsuite_fails_closed(evidence, xml):
    root, _, report = evidence
    (root / ".github").mkdir()
    (root / ".github/tests-ran-allowlist.txt").write_text("")
    target = root / "bad-structure.xml"
    target.write_text(xml)
    script = Path(__file__).resolve().parents[1] / "scripts/ci/check_tests_ran.py"
    result = subprocess.run(
        [sys.executable, str(script), "--junit", str(report()), str(target)],
        cwd=root, capture_output=True, text=True,
        env={**os.environ, "GITHUB_EVENT_NAME": "push", "GITHUB_EVENT_PATH": "",
             "GITHUB_STEP_SUMMARY": ""},
    )
    assert result.returncode == 1
    assert "missing or malformed JUnit: testcase must be directly under a testsuite" in result.stdout


def test_empty_report_fails_on_push_too(evidence):
    root, allowlist, report = evidence
    errors, _ = check(root, [report(cases=())], [], allowlist)
    assert any("zero executed tests" in error for error in errors)
    errors, _ = check(root, [], [], allowlist)
    assert errors == ["missing JUnit reports"]


def test_allowlist_cannot_hide_missing_reports(evidence):
    root, allowlist, _ = evidence
    allowlist.write_text("tests/test_a.py # needs a host\n")
    errors, _ = check(root, [root / "missing.xml"], ["tests/test_a.py"], allowlist)
    assert any("missing or malformed" in error for error in errors)


def test_resolve_absolute_and_relative_paths(evidence):
    root, allowlist, report = evidence
    errors, summary = check(root, [report(cases=((str(root / "tests/test_a.py"), "passed"),
                                               ("./tests/test_a.py", "passed")))],
                            ["tests/test_a.py"], allowlist)
    assert errors == []
    assert "tests/test_a.py | 2 | executed" in summary


@pytest.fixture
def changed_repo(tmp_path):
    def git(*args):
        return subprocess.check_output(["git", "-c", "user.name=Fixture",
                                        "-c", "user.email=fixture@example.invalid", *args],
                                       cwd=tmp_path, text=True).strip()
    git("init", "-q")
    for name in ("test_deleted.py", "test_old.py", "test_modified.py"):
        (tmp_path / name).write_text("# initial\n")
    git("add", ".")
    git("commit", "-qm", "base")
    base = git("rev-parse", "HEAD")
    (tmp_path / "test_deleted.py").unlink()
    git("mv", "test_old.py", "renamed_test.py")
    (tmp_path / "test_modified.py").write_text("# changed\n")
    (tmp_path / "test_added.py").write_text("# added\n")
    (tmp_path / "helper.py").write_text("# helper\n")
    git("add", ".")
    git("commit", "-qm", "head")
    return tmp_path, base, git("rev-parse", "HEAD")


@pytest.mark.parametrize("event", ["pull_request", "merge_group"])
def test_deleted_and_renamed_files(changed_repo, event):
    root, base, head = changed_repo
    payload = {"pull_request": {"base": {"sha": base}, "head": {"sha": head}},
               "merge_group": {"base_sha": base, "head_sha": head}}
    assert set(changed_tests(event, payload, root)) == {
        "renamed_test.py", "test_added.py", "test_modified.py"}


@pytest.mark.parametrize("event", ["push", "workflow_dispatch"])
def test_push_has_no_changed_file_check(tmp_path, event):
    assert changed_tests(event, {}, tmp_path) == []


@pytest.mark.parametrize("path,selected", [
    ("bench/instruments/compaction_probe/tests_material_v4.py", True),
    ("bench/instruments/compaction_probe/tests_compaction_probe_b.py", True),
    ("bench/instruments/compaction_probe/testprobe.py", True),
    ("bench/instruments/compaction_probe/helper.py", False),
    ("bench/instruments/compaction_probe/nested/tests_material_v4.py", False),
    ("bench/other/tests_material_v4.py", False),
    ("tests/test_regular.py", True),
    ("tests/regular_test.py", True),
])
def test_changed_test_selection_and_missing_evidence(evidence, monkeypatch, path, selected):
    root, allowlist, report = evidence
    monkeypatch.setattr(subprocess, "check_output", lambda *args, **kwargs: (path + "\0").encode())
    changed = changed_tests("merge_group", {"merge_group": {"base_sha": "base", "head_sha": "head"}}, root)
    assert changed == ([path] if selected else [])
    errors, _ = check(root, [report()], changed, allowlist)
    assert errors == ([f"{path}: not collected by any CI job"] if selected else [])


def test_compaction_probe_selection_matches_workflow_glob(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    workflow = (root / ".github/workflows/ci.yml").read_text()
    directory, pattern = re.findall(
        r'if \[\[ "\$root" == (bench/instruments/compaction_probe) \]\]; '
        r'then targets=\("\$root"/([^;)]+)\); fi', workflow,
    )[0]
    paths = sorted(p.relative_to(root).as_posix() for p in (root / directory).iterdir() if p.is_file())
    monkeypatch.setattr(subprocess, "check_output", lambda *args, **kwargs: "\0".join(paths).encode())
    selected = changed_tests("merge_group", {"merge_group": {"base_sha": "base", "head_sha": "head"}}, root)
    expected = {p.relative_to(root).as_posix() for p in (root / directory).glob(pattern) if p.is_file()}
    assert expected
    assert set(selected) == expected


def test_cli_writes_summary_and_fails_closed(evidence):
    root, allowlist, report = evidence
    (root / ".github").mkdir()
    allowlist.rename(root / ".github/tests-ran-allowlist.txt")
    changed = root / "changed.txt"
    changed.write_text("bench/test_missing.py\n")
    summary = root / "summary.md"
    event = root / "event.json"
    event.write_text(json.dumps({}))
    script = Path(__file__).resolve().parents[1] / "scripts/ci/check_tests_ran.py"
    result = subprocess.run([sys.executable, str(script), "--junit", str(report()),
                             "--changed-files", str(changed)], cwd=root, capture_output=True,
                            text=True, env={**os.environ, "GITHUB_EVENT_NAME": "push",
                                            "GITHUB_EVENT_PATH": str(event),
                                            "GITHUB_STEP_SUMMARY": str(summary)})
    assert result.returncode == 1
    assert "bench/test_missing.py: not collected by any CI job" in summary.read_text()


def test_workflow_requires_every_report_without_changing_test_selection():
    workflow = (Path(__file__).resolve().parents[1] / ".github/workflows/ci.yml").read_text()
    assert 'tests/ --collect-only -q -m "not latency" --ignore=tests/test_embedding_privacy.py' in workflow
    assert '-q -m "not latency" -n 4 -o junit_family=xunit1' in workflow
    assert 'tests/test_embedding_privacy.py -q -m "not latency" -o junit_family=xunit1' in workflow
    assert "needs: [test, test-312, test-313, test-314, bench-test-root]" in workflow
    assert "reports/test-{3.11,3.12,3.13,3.14}/shard-{1,2,3,4}/main.xml" in workflow
    assert "reports/test-{3.11,3.12,3.13,3.14}/privacy.xml" in workflow
    assert "shard-{n}/collection.txt" in workflow
    assert "if-no-files-found: error" in workflow
