"""Synthetic GLM route regressions; no network or real credentials."""
import contextlib
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from bench.fmp import common, delivery, judges, models, run, score

SPEC = "glm:glm-5.3"
HOST = "api.z.ai"


def test_glm_spec():
    assert models.parse_spec(SPEC) == ("glm", "glm-5.3", None)
    assert models.harness(SPEC)


def test_glm_empty_model():
    with pytest.raises(ValueError, match="invalid model spec"):
        models.parse_spec("glm:")


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("GLM_API_KEY", uuid.uuid4().hex)
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    result = models.Backends(tmp_path / "output", models.Quota(tmp_path / "quota.json"), [SPEC])
    yield result
    result.close()


def reply(cmd, **kwargs):
    return subprocess.CompletedProcess(cmd, 0, "\n".join(map(json.dumps, [
        {"type": "assistant", "message": {"model": "glm-5.3", "content": []}},
        {"type": "result", "subtype": "success", "is_error": False, "result": "CORRECT",
         "modelUsage": {"glm-5.3": {}}, "usage": {"input_tokens": 10, "output_tokens": 1},
         "stop_reason": "end_turn"}])), "")


def test_glm_child_environment(client, monkeypatch):
    backend, _ = client.backend(SPEC)
    for name in ("CLAUDECODE", "CLAUDE_CODE_ENTRYPOINT", "CLAUDE_CODE_OAUTH_TOKEN", "LOGNAME", "SHELL",
                 "ANTHROPIC_API_KEY", "OTHER_SECRET"):
        monkeypatch.setenv(name, "must-not-inherit")
    seen = []

    def runner(cmd, **kwargs):
        env = kwargs["env"]
        assert set(env) <= {"HOME", "USER", "PATH", "LANG", "TMPDIR", "CLAUDE_CONFIG_DIR",
                            "ANTHROPIC_BASE_URL", "ANTHROPIC_AUTH_TOKEN"}
        assert env["ANTHROPIC_AUTH_TOKEN"] == os.environ["GLM_API_KEY"]
        assert env["ANTHROPIC_BASE_URL"] == "https://api.z.ai/api/anthropic"
        assert not any(k == "CLAUDECODE" or k.startswith("CLAUDE_CODE_") for k in env)
        assert json.loads(Path(env["CLAUDE_CONFIG_DIR"], "settings.json").read_text()) == {}
        assert cmd[cmd.index("--effort") + 1] == "low"
        assert cmd[cmd.index("--model") + 1] == "glm-5.3"
        seen.append(env)
        return reply(cmd)

    backend.runner = runner
    records = []
    client.call(SPEC, [{"role": "user", "content": "Synthetic?"}], records)
    assert "ANTHROPIC_AUTH_TOKEN" not in seen[0]  # scrub the transient env after return
    assert "ANTHROPIC_AUTH_TOKEN" not in os.environ
    assert records[0]["backend"] == "glm" and records[0]["endpoint_host"] == HOST


@pytest.mark.parametrize("key", [None, ""])
def test_glm_missing_key_before_subprocess(tmp_path, monkeypatch, key):
    if key is None:
        monkeypatch.delenv("GLM_API_KEY", raising=False)
    else:
        monkeypatch.setenv("GLM_API_KEY", key)
    runner = Mock(side_effect=AssertionError("subprocess must not run"))
    monkeypatch.setattr(models.subprocess, "run", runner)
    with pytest.raises(ValueError, match="^GLM_API_KEY is not set$"):
        client = models.Backends(tmp_path, models.Quota(tmp_path / "quota.json"), [SPEC, models.CODEX_DEFAULT_JUDGE])
        try:
            client.call(SPEC, [{"role": "user", "content": "Synthetic?"}], [])
        finally:
            client.close()
    runner.assert_not_called()


@pytest.mark.parametrize("failure", [False, True])
def test_glm_fake_executable_judge_and_leak(client, tmp_path, monkeypatch, failure, capsys):
    binary = tmp_path / "claude"
    events = ([{"type": "result", "is_error": True, "api_error_status": 429,
                "result": "HTTP 429 rate-limit"}] if failure else [
        {"type": "assistant", "message": {"model": "glm-5.3", "content": []}},
        {"type": "result", "subtype": "success", "result": '{"label":"correct"}',
         "modelUsage": {"glm-5.3": {}}, "usage": {"input_tokens": 8, "output_tokens": 2}}])
    binary.write_text("#!" + sys.executable + "\nimport sys\nsys.stdin.read()\nprint(" +
                      repr("\n".join(map(json.dumps, events))) + ")\n" + ("sys.exit(1)\n" if failure else ""))
    binary.chmod(0o700)
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ.get("PATH", ""))
    q = SimpleNamespace(dataset="locomo", question="What is 2 + 2?", gold="4")
    records = []
    backend, _ = client.backend(SPEC)
    if failure:
        primary = "codex:synthetic@low"
        original = client.call

        def routed(spec, *args, **kwargs):
            return {"content": "CORRECT"} if spec == primary else original(spec, *args, **kwargs)

        monkeypatch.setattr(client, "call", routed)
        result, _ = judges.judge(q, "4", client, records, models=[primary], agreement_model=SPEC)
        assert result[primary]["correct"] is True and result[SPEC]["status"] == "not_run"
        assert client.quota.state["glm"]["stopped"] and not client.quota.state["claude"]["stopped"]
        with pytest.raises(models.BudgetStopped):
            original(SPEC, [{"role": "user", "content": "Synthetic?"}], [])
        assert score.quota(records)["glm_usage_limit_errors"] == 1
    else:
        result, _ = judges.judge(q, "4", client, records, models=[SPEC])
        assert result[SPEC]["correct"] is True and result[SPEC]["verdict"] == "valid"
    output = tmp_path / "output"
    (output / "result.json").write_text(json.dumps([result, records]))
    logs = capsys.readouterr()
    key = os.environ["GLM_API_KEY"].encode()
    assert all(key not in p.read_bytes() for p in output.rglob("*") if p.is_file())
    assert key not in (logs.out + logs.err).encode()
    assert backend.binary == "claude"


@pytest.mark.parametrize("mode", ["single", "full_context", "agentic"])
def test_glm_reader(client, monkeypatch, mode):
    backend, _ = client.backend(SPEC)
    backend.runner = reply
    monkeypatch.setattr(delivery, "question_clock", lambda date: contextlib.nullcontext())
    monkeypatch.setattr(delivery, "single", lambda *a: ("Synthetic history.", "{}"))
    q = SimpleNamespace(qid="synthetic", qtype="1", question="Synthetic?", question_date="2025-01-01",
                        sessions=[{"date": "2025-01-01", "turns": [{"role": "user", "content": "Synthetic."}]}])
    assert client.context_length(SPEC) == 128_000  # conservative harness admission cap
    row = delivery.answer(q, mode, client, SPEC, server={"question": {}})
    assert row["status"] == "ok" and row["hypothesis"] == "CORRECT"


def test_glm_estimate_and_manifest(client, tmp_path):
    args = run.parser().parse_args(["judge", "--split-label", "synthetic", "--store-root", str(tmp_path),
                                   "--output", str(tmp_path / "output"), "--judge", SPEC,
                                   "--agreement-judge", "none"])
    estimate = run.estimate(args, 1)
    assert estimate["plans"]["glm_coding_plan"]["specs"] == [SPEC]
    output = args.output
    q = SimpleNamespace(qid="synthetic", dataset="locomo", question="Synthetic?", gold="yes")
    (output / "run-manifest.json").write_text(json.dumps({"dataset_sha256": "d", "id_list_sha256": "i",
        "split_label": "synthetic", "question_ids": [q.qid], "prompts_sha256": run.prompt_hashes(), **run.switches(args),
        "harness_commit": common.commit(Path(run.__file__).resolve().parents[2]),
        "lcmx_commit": common.commit(args.lcmx_repo),
        "harness_tree_sha256": common.sha(b"".join(p.read_bytes() for p in sorted(Path(run.__file__).parent.glob("*.py"))))}))
    (output / "answers.jsonl").write_text(json.dumps({"qid": q.qid, "status": "ok", "hypothesis": "yes"}) + "\n")
    backend, _ = client.backend(SPEC)
    backend.runner = reply
    run.run_jobs([run.run(args, client, [q], "d", "i", {}, None)], 1)
    manifest = json.loads((output / "run-manifest.json").read_text())
    assert manifest["judges"] == [SPEC] and manifest["endpoint_hosts"][SPEC] == HOST
    assert os.environ["GLM_API_KEY"] not in json.dumps(manifest)


def test_run_sh_rejects_old_python(tmp_path):
    binary = tmp_path / "old-python"
    binary.write_text('#!/bin/sh\nif [ "$1" = "-c" ]; then exit 1; fi\nexit 0\n')
    binary.chmod(0o700)
    script = Path(run.__file__).with_name("run.sh")
    proc = subprocess.run(["bash", str(script), "--help"], env={**os.environ, "FMP_PYTHON": str(binary)},
                          capture_output=True, text=True)
    assert proc.returncode != 0 and "FMP_PYTHON requires Python >= 3.11" in proc.stderr
