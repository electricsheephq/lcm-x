import hashlib
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "run_candidate.sh"
REPO = Path(__file__).resolve().parents[3]
HELPER = REPO / "scripts/dev/ci_step.sh"
PATH = os.pathsep.join([str(Path(sys.executable).parent), os.environ["PATH"]])
NEW_TEST = '''
def test_new():
    import os
    import subprocess
    from pathlib import Path
    assert os.environ["FIXTURE_ENV"] == "loaded"
    home = Path(os.environ["HOME"]).resolve()
    assert home.name.startswith("home-") and Path.cwd().resolve() not in home.parents
    status = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=True)
    assert status.stdout == ""
    assert int(Path("tiny.py").read_text().split()[-1]) == 2
'''
# The fixture's copy of CI's stub step; like the real one, it writes the gitignored agent/.
CI = """jobs:
  test:
    steps:
      - name: Stub hermes-agent ABC
        run: |
          mkdir -p agent
          echo stubbed > agent/stub.txt
"""


def test_run_candidate(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    home = tmp_path / "isolated-home"
    home.mkdir()
    # A ruff stand-in that records its arguments, so the test does not need ruff installed.
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    ruff = bin_dir / "ruff"
    ruff.write_text('#!/bin/sh\nprintf "%s\\n" "$@" > "$RUFF_ARGS"\n')
    ruff.chmod(0o755)
    ruff_args = tmp_path / "ruff-args.txt"
    env = dict(os.environ, HOME=str(home), XDG_CONFIG_HOME=str(home / ".config"),
               GIT_CONFIG_NOSYSTEM="1", RUFF_ARGS=str(ruff_args),
               PATH=os.pathsep.join([str(bin_dir), PATH]))
    env.pop("FIXTURE_UNSET", None)

    def git(*args):
        return subprocess.check_output(["git", "-C", str(repo), *args], env=env)

    def make_patch(name):
        # The README's command over the staged tree, so renames and deletions are included.
        git("add", "-A")
        path = tmp_path / name
        path.write_bytes(git("-c", "diff.noprefix=false", "diff", "--cached", "--binary"))
        git("reset", "-q", "--hard")
        return path

    git("init", "-q")
    (repo / "tests").mkdir()
    (repo / ".github/workflows").mkdir(parents=True)
    (repo / ".github/workflows/ci.yml").write_text(CI)
    (repo / ".gitignore").write_text("__pycache__/\nagent/\n")
    (repo / "notes.txt").write_text("old\n")
    (repo / "tests/test_embedding_privacy.py").write_text("def test_alone():\n    assert True\n")
    module = repo / "tiny.py"
    module.write_text("VALUE = 1\n")
    test = repo / "tests/test_tiny.py"
    test.write_text("def test_old():\n    assert True\n")
    git("add", ".")
    git("-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
        "-c", "commit.gpgsign=false", "commit", "--no-verify", "-qm", "base")
    base = git("rev-parse", "HEAD").decode().strip()
    module.write_text("VALUE = 2\n")
    test.write_text(test.read_text() + textwrap.dedent(NEW_TEST))
    # NEW_TEST sees a dirty tree if the rename's deletion is left out of the commit.
    git("mv", "notes.txt", "notes.md")
    patch = make_patch("candidate.patch")
    (repo / "notes.txt").write_text("new\n")
    docs = make_patch("docs.patch")
    git("rm", "-q", "tests/test_embedding_privacy.py")
    no_privacy = make_patch("no-privacy.patch")
    # The runner sources the env file even when it expands an unset variable.
    env_file = tmp_path / "env.sh"
    env_file.write_text('export FIXTURE_ENV="loaded$FIXTURE_UNSET"\n')
    root = tmp_path / "runs with space"

    def run(label, *extra, patch_file=patch, stub=False):
        return subprocess.run(["bash", str(SCRIPT), "--base", base, "--patch", str(patch_file),
                               "--root", str(root), "--label", label, "--no-fetch",
                               *([] if stub else ["--no-stub"]),
                               "--env-file", str(env_file), *extra], cwd=repo, env=env,
                              text=True, capture_output=True, timeout=30)

    result = run("first", "--focused", "tests/", "--red", "tests/", "--full",
                 "--workers-focused", "1", "--workers-full", "1")
    assert result.returncode == 0, result.stdout + result.stderr
    summary_file = next(root.glob("run-first-*/summary.json"))
    data = json.loads(summary_file.read_text())
    assert data["base"] == base and data["label"] == "first" and data["stamp"]
    assert data["patch_sha256"] == hashlib.sha256(patch.read_bytes()).hexdigest()
    assert data["python"] and data["sqlite"]
    phases = data["phases"]
    assert {"ruff", "green-focused", "red-on-base", "green-full", "green-privacy"} == set(phases)
    assert phases["green-focused"]["exit_code"] == phases["green-full"]["exit_code"] == 0
    assert phases["green-privacy"]["exit_code"] == 0 and "1 passed" in phases["green-privacy"]["final_line"]
    assert "2 passed" in phases["green-full"]["final_line"]
    assert phases["red-on-base"]["exit_code"] == 1
    assert "FAILED tests/test_tiny.py::test_new" in result.stdout
    assert phases["ruff"]["exit_code"] == 0
    assert ruff_args.read_text().split() == ["check", "--", "tests/test_tiny.py", "tiny.py"]
    assert all(p["command"] and "final_line" in p for p in phases.values())
    assert str(tmp_path) not in summary_file.read_text()
    assert result.stdout.count("git worktree remove ") == 2
    before = set(root.iterdir())
    # This run takes the stub step from the candidate tree's ci.yml.
    stubbed = run("second", stub=True)
    assert stubbed.returncode == 0, stubbed.stdout + stubbed.stderr
    assert before < set(root.iterdir()) and summary_file.exists()
    assert (next(root.glob("wt-second-green-*")) / "agent/stub.txt").read_text() == "stubbed\n"
    # No .py file changed (an empty array under bash 3.2's set -u), then a red run with no tests/ hunks.
    no_tests = run("docs", "--red", "tests/", patch_file=docs)
    assert no_tests.returncode == 2 and "NO_TEST_HUNKS" in no_tests.stderr
    assert "changed_py:\n" in no_tests.stdout and no_tests.stdout.count("git worktree remove ") == 2
    # CI runs the privacy file even when a candidate deletes it.
    deleted = run("noprivacy", "--full", "--workers-full", "1", patch_file=no_privacy)
    assert deleted.returncode == 2 and "PYTEST_SETUP_FAILED" in deleted.stderr
    deleted_phases = json.loads(next(root.glob("run-noprivacy-*/summary.json")).read_text())["phases"]
    assert deleted_phases["green-full"]["exit_code"] == 0
    assert deleted_phases["green-privacy"]["exit_code"] == 4
    blank = run("blank", "--focused", " ")
    assert blank.returncode == 2 and "BAD_ARGUMENT" in blank.stderr
    missing = run("missing", "--env-file", str(tmp_path / "absent"))
    assert missing.returncode == 2 and "NO_ENV" in missing.stderr
    failing_env = tmp_path / "failing-env.sh"
    failing_env.write_text("false\n")
    failing = run("failing", "--env-file", str(failing_env))
    assert failing.returncode == 2 and "NO_ENV" in failing.stderr
    assert "NO_PATCH" in run("nopatch", patch_file=tmp_path / "absent.patch").stderr
    patch.write_text(patch.read_text().replace("VALUE = 1", "VALUE = 99"))
    bad = run("bad")
    assert bad.returncode == 2 and "APPLY_FAILED" in bad.stderr


def step_script(path, name):
    return subprocess.run(["bash", "-c", '. "$0" && step_script "$1" "$2"', str(HELPER), str(path), name],
                          env=dict(os.environ, PATH=PATH), text=True, capture_output=True, timeout=30)


def test_ci_step_reads_ci_and_fails_closed(tmp_path):
    ci = REPO / ".github/workflows/ci.yml"
    stub = step_script(ci, "Stub hermes-agent ABC")
    assert stub.returncode == 0, stub.stderr
    assert "class ContextEngine" in stub.stdout and not stub.stdout.startswith(" ")
    missing = step_script(ci, "No such step")
    assert missing.returncode != 0 and "ci_step: no run block" in missing.stderr
    workflow = tmp_path / "ci.yml"
    workflow.write_text("jobs:\n  test:\n    steps:\n      - name: Plain\n        run: echo a\n"
                        "      - name: Block\n        run: |\n          echo b\n\n          echo c\n"
                        "        env:\n          X: 1\n  lint:\n    steps: []\n")
    assert step_script(workflow, "Plain").stdout == "echo a\n"
    assert step_script(workflow, "Block").stdout == "echo b\n\necho c\n"
    # A "- name:" line inside a run block does not start a step.
    workflow.write_text("jobs:\n  test:\n    steps:\n      - name: Write\n        run: |\n"
                        "          cat <<'EOF'\n          - name: Step\n            run: echo wrong\n"
                        "          EOF\n      - name: Step\n        run: echo right\n")
    assert step_script(workflow, "Step").stdout == "echo right\n"
    assert step_script(workflow, "Write").stdout == "cat <<'EOF'\n- name: Step\n  run: echo wrong\nEOF\n"
    for run in ("|-\n          echo a", "| # note\n          echo a", ">\n          echo a",
                '"echo a"', "echo a\n          continued", "|\n        env: {}"):
        workflow.write_text(f"jobs:\n  test:\n    steps:\n      - name: Step\n        run: {run}\n")
        result = step_script(workflow, "Step")
        assert result.returncode != 0 and "ci_step: unsupported run form" in result.stderr, run
    workflow.write_text("jobs:\n  test:\n    runs-on: x\n")
    assert "ci_step: no steps" in step_script(workflow, "Step").stderr
    workflow.write_text("jobs:\n  lint:\n    steps: []\n")
    assert "ci_step: no test job" in step_script(workflow, "Step").stderr
