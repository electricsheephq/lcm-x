"""The host's reviewed dependency overrides must reach its installer (#1919)."""
import json
from pathlib import Path

import pytest

from bench.instruments.reliability import ci


@pytest.mark.parametrize("overrides", [None, [], ["Pillow==13.0.0", "spacy-curated-transformers; sys_platform == 'never'"]],
                         ids=["undeclared", "empty", "declared"])
def test_prep_preserves_host_overrides_and_pins(tmp_path, monkeypatch, overrides):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"hosts": {"fixture": {
        "repo": "example/host", "sha": "pinned", "python_version": "3.14",
        "extras": "acp", "pins": ["agent-client-protocol==0.9.0", "pip==24.0"],
        "hermes_version": "0.0.0",
    }}}))
    monkeypatch.setattr(ci, "CI_HOSTS", manifest)
    src = tmp_path / "fixture" / "src"
    src.mkdir(parents=True)
    project = '[project]\nname = "fixture"\n'
    if overrides is not None:
        project += '[tool.uv]\noverride-dependencies = ' + json.dumps(overrides) + '\n'
    (src / "pyproject.toml").write_text(project)
    commands = []
    monkeypatch.setattr(ci.subprocess, "run", lambda cmd, **kw: commands.append(cmd))

    result = ci.prep("fixture", tmp_path, tmp_path / "hosts.json")

    install = commands[-1]
    assert install[:3] == ["uv", "pip", "install"]
    assert "agent-client-protocol==0.9.0" in install and "pip==24.0" in install
    if overrides:
        forwarded = Path(install[install.index("--overrides") + 1])
        assert forwarded.parent == tmp_path / "fixture"  # Keep the pinned source tree clean.
        assert forwarded.read_text().splitlines() == overrides
    else:
        assert "--overrides" not in install
        assert not (tmp_path / "fixture" / "overrides.txt").exists()
        assert install == ["uv", "pip", "install", "-q", "-p",
                           str(tmp_path / "fixture" / "venv" / "bin" / "python"),
                           "-e", f"{src}[acp]", "agent-client-protocol==0.9.0", "pip==24.0"]
    assert (src / "pyproject.toml").read_text() == project
    assert result == json.loads((tmp_path / "hosts.json").read_text())
    assert result["hosts"]["fixture"]["sha"] == "pinned"
