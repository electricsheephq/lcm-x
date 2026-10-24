"""Executed only in an isolated prepared Hermes interpreter; never system Hermes."""
import ast
import json
import os
from pathlib import Path
import sys
import textwrap

from hermes_cli import plugin_catalog as catalog
from hermes_cli import plugins_cmd as pc
from hermes_cli import plugins_cmd_catalog as cc
from tools import plugin_guard as guard

phase, config, receipt, expected, probe = sys.argv[1:]
entries = json.loads(Path(config).read_text())
home = Path(os.environ["HERMES_HOME"])
home.mkdir(parents=True)
events = []
original = guard.scan_plugin


def observe(*args, **kwargs):
    result = original(*args, **kwargs)
    findings = sorted({(f.pattern_id, f.severity, f.file) for f in result.findings})
    events.append(dict(verdict=result.verdict, scanner=guard.PLUGIN_SCANNER_VERSION, findings=findings))
    assert result.verdict != "dangerous"
    medium = [list(f) for f in findings if f[1] in {"medium", "high", "critical"}]
    baseline = json.loads(Path(expected).read_text()) if phase == "candidate" else [
        ["python_subprocess", "medium", "inactive_record.py"],
        ["python_subprocess", "medium", "runtime_identity.py"],
        ["unpinned_pip_install", "medium", "diagnostics.py"]]
    assert medium == baseline, (medium, baseline)
    return result


def cache(label):
    raw = {"entries": [entries[label]], "removed": []}
    target = home / "cache/plugin-catalog.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(raw))
    return catalog.entry_from_mapping(entries[label], "local-copy")


assert pc._scan_on_install_enabled() is True
guard.scan_plugin = observe  # Observe the unchanged scanner; preserve its return value.
try:
    label = "candidate" if phase == "candidate" else "baseline"
    entry = cache(label)
    assert entry is not None
    target, manifest, name = cc.install_catalog_entry(entry, force=False)
    revision = pc._read_install_metadata()[name]["revision"]
    assert revision == entries[label]["sha"]
    assert events
    if phase == "consent":
        cache("candidate")
        try:
            cc.repin_catalog_plugin(target, cc.read_catalog_sidecar(target), interactive=False)
        except cc.RepinConsentRequired:
            assert pc._read_install_metadata()[name]["revision"] == revision
            assert len(events) == 1  # Consent refusal precedes the scan/transaction.
        else:
            raise AssertionError("expected RepinConsentRequired for new tool consent")
    else:
        (home / "config.yaml").write_text("plugins:\n  enabled: [hermes-lcm-x]\ncontext:\n  engine: lcm-x\n")
        tree = ast.parse(Path(probe).read_text())
        code = next(n.value.args[0].value for n in tree.body if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == "_PROBE" for t in n.targets))
        namespace = {}
        exec(textwrap.dedent(code), namespace)
        assert namespace["selected"].name == "lcm-x"
        import inspect
        assert Path(inspect.getfile(type(namespace["selected"]))).resolve().is_relative_to(target.resolve())
        from importlib import import_module
        package = type(namespace["selected"]).__module__.rsplit(".", 1)[0]
        assert import_module(package + ".guidance").get_recall_policy()
finally:
    Path(receipt).write_text(json.dumps(dict(phase=phase, scanner_on=True, events=events), indent=2) + "\n")
