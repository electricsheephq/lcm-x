"""Regression for PR 1689 sandbox."""
import os
import subprocess
import pytest
from .bot1_support import S2


@pytest.mark.xfail(reason="Narrowed profile failed the one-call H smoke; original profile restored per spec", strict=True)
def test_h_sandbox_account_configuration_readonly(tmp_path):
    fake = tmp_path / "bin"
    fake.mkdir()
    exe = fake / "sandbox-exec"
    exe.write_text("#!/bin/sh\nexit 0\n")
    exe.chmod(0o755)
    account = tmp_path / "account"
    output, temp = tmp_path / "output", tmp_path / "tmp"
    temp.mkdir()
    env = dict(os.environ, PATH=str(fake) + os.pathsep + os.environ["PATH"], TMPDIR=str(temp), TRACK_S_OUT=str(output),
        HERMES_SRC=str(tmp_path / "host/src"), C1_BUDGET_ROOT=str(tmp_path / "budget"), CODEX_HOME=str(account))
    subprocess.run(["bash", str(S2 / "launch_h.sh")], env=env, check=True, capture_output=True)
    profile = (output / "h-isolation.sb").read_text()
    grants = "\n".join(line for line in profile.splitlines() if line.startswith("(allow file-write*"))
    assert '(subpath "' + str(account) + '")' not in grants
    for name in ("config.toml", "AGENTS.md", "agents", "skills", "prompts", "rules", "plugins"):
        assert str(account / name) in profile
    assert str(account / "sessions") in profile and str(account / "log") in profile
    assert '(literal "' + str(account / "auth.json") + '")' in profile
