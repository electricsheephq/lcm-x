"""Resolve config through the plugin's public path in an unmodified host interpreter (#1919)."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import textwrap

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
HERMES_PYTHON = os.environ.get("LCM_REAL_HERMES_PYTHON")
pytestmark = pytest.mark.skipif(not HERMES_PYTHON, reason="LCM_REAL_HERMES_PYTHON is unset")

_PROBE = textwrap.dedent('''
    import json, os, sys
    from pathlib import Path

    try:
        import yaml
    except ImportError:
        yaml = None
    try:
        import ruamel.yaml as ruamel_yaml
    except ImportError:
        ruamel_yaml = None
    import config

    ran = set()
    def observe(frame, event, arg):
        if event != "call":
            return
        module, name = frame.f_globals.get("__name__"), frame.f_code.co_name
        if (module, name) == ("yaml", "safe_load"):
            ran.add("PyYAML")
        elif (module, name) == ("ruamel.yaml.main", "load"):
            ran.add("ruamel")
        elif (module, name) == ("config", "_parse_yaml_fallback_scalar"):
            ran.add("line parser")

    def resolve():
        resolved = config.LCMConfig.from_env()
        return {
            "context_threshold": resolved.context_threshold,
            "context_source": resolved.config_sources["context_threshold"],
            "model_thresholds": resolved.model_thresholds,
            "codex_gpt55_autoraise_enabled": resolved.codex_gpt55_autoraise_enabled,
            "embedding_query_prefix": resolved.embedding_query_prefix,
            "embedding_document_prefix": resolved.embedding_document_prefix,
            "summary_timeout_ms": resolved.summary_timeout_ms,
            "ignored_config_yaml_lcm_keys": resolved.ignored_config_yaml_lcm_keys,
        }

    sys.setprofile(observe)
    explicit = resolve()
    parsed = config._load_hermes_config_yaml()
    path = Path(os.environ["HERMES_HOME"]) / "config.yaml"
    path.write_text(path.read_text().replace("  context_threshold: 0.63\\n", ""))
    inherited = resolve()  # Exercise compression.enabled without the explicit LCM override.
    sys.setprofile(None)
    print(json.dumps({
        "libraries": {"PyYAML": yaml is not None, "ruamel": ruamel_yaml is not None},
        "loaders": sorted(ran), "explicit": explicit, "inherited": inherited,
        "flow_list": parsed["lcm"]["unsupported_modes"],
    }))
''')


@pytest.fixture
def yaml_config(tmp_path, request):
    enabled, autoraise = request.param
    path = tmp_path / "config.yaml"
    path.write_text(textwrap.dedent(f'''\
        compression:
          enabled: {enabled}
          threshold: 0.91
          codex_gpt55_autoraise: {autoraise}
        auxiliary:
          compression:
            timeout: 12.5
        lcm:
          context_threshold: 0.63
          model_thresholds:
            'single # model': 0.42
            "double # model": 1
          embedding_query_prefix: "true"
          embedding_document_prefix: "1.10"
          unsupported_modes: ['single # tag', "double # tag", "true", "1.10"]
        '''))
    return path


@pytest.mark.parametrize(
    "yaml_config, expected_autoraise, expected_threshold, expected_source",
    [
        (("off", "no"), False, 0.75, "default"),
        (("false", "yes"), True, 0.75, "default"),
        (("yes", "false"), False, 0.91, "config_yaml:compression.threshold"),
        (("no", "off"), False, 0.75, "default"),
    ],
    indirect=["yaml_config"],
)
def test_real_host_resolved_config(yaml_config, expected_autoraise, expected_threshold, expected_source):
    src = os.environ.get("LCM_REAL_HERMES_SRC", "")
    completed = subprocess.run(
        [HERMES_PYTHON, "-c", _PROBE], cwd=yaml_config.parent,
        env={
            "HOME": str(yaml_config.parent), "HERMES_HOME": str(yaml_config.parent),
            "PATH": "/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE": "1",
            "HERMES_DISABLE_LAZY_INSTALLS": "1",
            "PYTHONPATH": os.pathsep.join(filter(None, (str(REPO_ROOT), src))),
        },
        capture_output=True, text=True, timeout=60, check=False,
    )
    assert completed.returncode == 0, completed.stderr[-4000:]
    result = json.loads(completed.stdout.strip().splitlines()[-1])
    print(json.dumps(result, sort_keys=True))
    libraries = result["libraries"]
    expected_loader = "PyYAML" if libraries["PyYAML"] else "ruamel" if libraries["ruamel"] else "line parser"
    assert result["loaders"] == [expected_loader], result
    # Hand-written YAML 1.1 expectations; no parser or resolver computes this oracle.
    assert result["flow_list"] == ["single # tag", "double # tag", "true", "1.10"]
    expected = {
        "context_threshold": 0.63,
        "context_source": "config_yaml:lcm.context_threshold",
        "model_thresholds": {"single # model": 0.42, "double # model": 1.0},
        "codex_gpt55_autoraise_enabled": expected_autoraise,
        "embedding_query_prefix": "true",
        "embedding_document_prefix": "1.10",
        "summary_timeout_ms": 12500,
        "ignored_config_yaml_lcm_keys": ["unsupported_modes"],
    }
    assert result["explicit"] == expected
    assert result["inherited"] == {
        **expected, "context_threshold": expected_threshold, "context_source": expected_source,
    }
