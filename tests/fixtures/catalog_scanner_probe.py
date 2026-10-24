"""Executed only in an isolated prepared Hermes interpreter; never system Hermes."""
import ast
from collections import Counter
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
home.mkdir(parents=True, exist_ok=True)
events = []
details = {}
original = guard.scan_plugin


def observe(*args, **kwargs):
    result = original(*args, **kwargs)
    counts = Counter((f.pattern_id, f.severity, f.file) for f in result.findings)
    findings = sorted((*key, count) for key, count in counts.items())
    events.append(dict(verdict=result.verdict, scanner=guard.PLUGIN_SCANNER_VERSION, findings=findings))
    assert result.verdict != "dangerous"
    medium = [list(f) for f in findings if f[1] in {"medium", "high", "critical"}]
    baseline = json.loads(Path(expected).read_text()) if phase == "candidate" or len(events) > 1 else [
        ["python_subprocess", "medium", "inactive_record.py", 1],
        ["python_subprocess", "medium", "runtime_identity.py", 1],
        ["unpinned_pip_install", "medium", "diagnostics.py", 1]]
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
    details["revision"] = revision
    assert events
    if phase == "consent":
        cache("candidate")
        try:
            cc.repin_catalog_plugin(target, cc.read_catalog_sidecar(target))
        except cc.RepinConsentRequired:
            assert pc._read_install_metadata()[name]["revision"] == revision
            assert len(events) in (1, 2)  # Ref host scans before consent; upstream previews first.
            details["expected_negative"] = "RepinConsentRequired"
            details["retained_revision"] = revision
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
        details.update(engine="lcm-x", engine_origin_in_payload=True)
finally:
    Path(receipt).write_text(json.dumps(dict(phase=phase, scanner_on=True, events=events, **details), indent=2) + "\n")
