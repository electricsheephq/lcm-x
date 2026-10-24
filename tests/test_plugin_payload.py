"""Payload contract, including mutations of imports, assets and shipped paths (#1946)."""
import ast
from collections import Counter
import hashlib
import importlib.util
import json
import re
from pathlib import Path
import subprocess
import sys
import textwrap
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def builder():
    path = ROOT / "scripts/build_plugin_payload.py"
    assert path.is_file(), "payload builder is required"
    spec = importlib.util.spec_from_file_location("payload_builder", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def commit(root):
    git(root, "add", ".")
    git(root, "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
        "commit", "--allow-empty", "-m", "fixture update")


@pytest.fixture
def source(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    files = {
        "__init__.py": "from . import core\n",
        "core.py": "from .package.child import VALUE\n",
        "package/__init__.py": "",
        "package/child.py": "VALUE = 1\n",
        "plugin.yaml": "name: fixture\n",
        "LICENSE": "fixture license\n",
        "skills/hermes-lcm/SKILL.md": "fixture skill\n",
    }
    for name, data in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(data)
    git(root, "init", "-b", "main")
    git(root, "add", ".")
    git(root, "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
        "commit", "-m", "fixture")
    return root


def build(builder, source, tmp_path, assets=None):
    allowlist = tmp_path / "assets.txt"
    allowlist.write_text(assets or "plugin.yaml\nLICENSE\nskills/hermes-lcm/**\n")
    return builder.build(str(source), tmp_path / "payload", tmp_path / "manifest.json", allowlist)


def test_import_closure_and_manifest(builder, source, tmp_path):
    result = build(builder, source, tmp_path)
    paths = [row["path"] for row in result["files"]]
    assert paths == sorted(paths)
    assert {"__init__.py", "core.py", "package/__init__.py", "package/child.py"} <= set(paths)
    assert result["source_sha"] == git(source, "rev-parse", "HEAD")
    assert result["builder_version"]
    assert result["tooling_sha"] == git(ROOT, "rev-parse", "HEAD")
    assert result["allowlist_sha256"] == hashlib.sha256((tmp_path / "assets.txt").read_bytes()).hexdigest()
    assert result["file_count"] == len(paths)
    assert result == json.loads((tmp_path / "manifest.json").read_text())
    for row in result["files"]:
        data = (tmp_path / "payload" / row["path"]).read_bytes()
        assert row["sha256"] == hashlib.sha256(data).hexdigest()
        assert row["bytes"] == len(data)
    assert result["total_bytes"] == sum(row["bytes"] for row in result["files"])


@pytest.mark.parametrize("text", ["from .missing import value", "from . import missing",
                                        "import hermes_lcm.missing"])
def test_unresolved_local_import_is_refused(builder, source, tmp_path, text):
    (source / "core.py").write_text(text)
    commit(source)
    with pytest.raises(ValueError, match="unresolved local import"):
        build(builder, source, tmp_path)
    assert not (tmp_path / "payload").exists()


@pytest.mark.parametrize("assets", ["plugin.yaml\nLICENSE\n", "plugin.yaml\nLICENSE\nmissing.md\n"])
def test_missing_or_unlisted_assets_refused(builder, source, tmp_path, assets):
    with pytest.raises(ValueError, match="asset"):
        build(builder, source, tmp_path, assets)


@pytest.mark.parametrize("prefix", ["tests", "bench", "scripts", ".github"])
def test_development_import_mutation_refused(builder, source, tmp_path, prefix):
    (source / prefix).mkdir()
    (source / prefix / "__init__.py").write_text("")
    (source / "core.py").write_text(f"from . import {prefix}" if prefix != ".github" else "")
    git(source, "add", ".")
    commit(source)
    assets = "plugin.yaml\nLICENSE\nskills/hermes-lcm/**\n" + prefix + "/**\n"
    with pytest.raises(ValueError, match="forbidden payload path"):
        build(builder, source, tmp_path, assets)


def test_external_symlink_refused(builder, source, tmp_path):
    outside = tmp_path / "outside.md"
    outside.write_text("outside")
    (source / "skills/hermes-lcm/escape.md").symlink_to(outside)
    git(source, "add", ".")
    commit(source)
    with pytest.raises(ValueError, match="symlink"):
        build(builder, source, tmp_path)


@pytest.mark.parametrize("limit,value", [("MAX_FILE_BYTES", 8), ("MAX_FILES", 2), ("MAX_TOTAL_BYTES", 20)])
def test_limits_refused_before_writing(builder, source, tmp_path, monkeypatch, limit, value):
    monkeypatch.setattr(builder, limit, value)
    with pytest.raises(ValueError, match="limit"):
        build(builder, source, tmp_path)
    assert not (tmp_path / "payload").exists()


def test_git_sha_source_matches_checkout(builder, source, tmp_path, monkeypatch):
    monkeypatch.chdir(source)
    result = build(builder, git(source, "rev-parse", "HEAD"), tmp_path)
    assert result["source_sha"] == git(source, "rev-parse", "HEAD")
    assert "package/child.py" in {row["path"] for row in result["files"]}


def test_checkout_drift_gate(builder, tmp_path):
    result = builder.build(str(ROOT), tmp_path / "payload", tmp_path / "manifest.json")
    paths = {row["path"] for row in result["files"]}
    # Re-derive using only the staged tree: a missing import cannot use the source checkout.
    builder.closure(tmp_path / "payload", paths, lambda p: (tmp_path / "payload" / p).read_bytes())
    assert result["file_count"] <= builder.MAX_FILES
    assert result["total_bytes"] <= builder.MAX_TOTAL_BYTES
    assert all(row["bytes"] <= builder.MAX_FILE_BYTES for row in result["files"])
    assert not any(p.split("/")[0] in {"tests", "bench", "scripts", ".github"} for p in paths)
    # Reviewed file-read sites from the #1946 measure lane: standalone config, skill
    # registration and guidance recall policy. engine/runtime_identity __file__ sites
    # only report provenance; bypass/host_uid_emit import host-owned modules.
    assert "config.py" in paths
    assert (tmp_path / "payload/skills/hermes-lcm/references/recall-policy.md").read_text().strip()
    assert {p.relative_to(ROOT).as_posix() for p in (ROOT / "skills/hermes-lcm").rglob("*") if p.is_file()} <= paths
    assert_reviewed_sites(tmp_path / "payload")


def reviewed_sites(payload):
    sites = []
    names = {"import_module", "__import__", "spec_from_file_location"}
    trees = {path: ast.parse(path.read_bytes()) for path in sorted(payload.rglob("*.py"))}
    def module(path):
        return path.relative_to(payload).with_suffix("").as_posix().replace("/", ".")
    roots = set()
    for path, tree in trees.items():
        for assignment in tree.body:
            if isinstance(assignment, (ast.Assign, ast.AnnAssign)) and assignment.value is not None:
                if any((isinstance(n, ast.Name) and n.id == "__file__") or
                       (isinstance(n, ast.Attribute) and n.attr == "__file__") for n in ast.walk(assignment.value)):
                    targets = assignment.targets if isinstance(assignment, ast.Assign) else [assignment.target]
                    roots.update(f"{module(path)}.{n.id}" for target in targets for n in ast.walk(target)
                                 if isinstance(n, ast.Name))
    for path, tree in trees.items():
        parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
        aliases = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                aliases.update((a.asname or a.name.split(".")[0], a.name if a.asname else a.name.split(".")[0])
                               for a in node.names)
            elif isinstance(node, ast.ImportFrom):
                prefix = list(path.parent.relative_to(payload).parts)
                prefix = prefix[:len(prefix) - node.level + 1] if node.level else []
                origin = ".".join([*prefix, node.module or ""]).strip(".")
                aliases.update((a.asname or a.name, (origin + "." + a.name).strip(".")) for a in node.names)

        def canonical(node):
            text = ast.unparse(node)
            first, dot, rest = text.partition(".")
            return aliases.get(first, first) + dot + rest

        for node in ast.walk(tree):
            if (isinstance(node, (ast.Name, ast.Attribute)) and isinstance(node.ctx, ast.Load)
                    and (canonical(node).split(".")[-1] in names | {"__file__"}
                         or canonical(node).startswith("importlib.resources")
                         or canonical(node) in roots or f"{module(path)}.{canonical(node)}" in roots)):
                # Include the enclosing expression so a changed asset path is drift too.
                expression = node
                while isinstance(parent := parents.get(expression), ast.expr) and not isinstance(
                        parent, (ast.Dict, ast.List, ast.Tuple, ast.Set)):
                    expression = parent
                sites.append([path.relative_to(payload).as_posix(), ast.unparse(expression)])
    return sorted(sites)


def assert_reviewed_sites(payload):
    expected = json.loads((ROOT / "tests/fixtures/payload-reviewed-sites.json").read_text())
    actual = reviewed_sites(payload)
    assert actual == expected, (actual, expected)


@pytest.mark.parametrize("site", ["__file__", "import_module(name)", "__import__(name)",
                                       "spec_from_file_location(name, path)", "importlib.resources.files(name)",
                                       "from importlib import resources as r; r.files(name)",
                                       "from importlib import import_module as im; im(name)",
                                       "Path(config.__file__)", "Path(sys.modules[__name__].__file__)",
                                       "list(map(import_module, names))", "_load = importlib.import_module",
                                       "functools.partial(importlib.util.spec_from_file_location, name)",
                                       "_load = __import__"])
def test_new_unreviewed_site_mutation_fails(builder, tmp_path, site):
    payload = tmp_path / "payload"
    builder.build(git(ROOT, "rev-parse", "HEAD"), payload, tmp_path / "manifest.json")
    assert_reviewed_sites(payload)
    with (payload / "__init__.py").open("a") as stream:
        stream.write("\n" + site + "\n")
    with pytest.raises(AssertionError):
        assert_reviewed_sites(payload)
    print(f"new reviewed-site mutation rejected: {site}")


@pytest.mark.parametrize("module,prefix", [("runtime_identity.py", ""),
                                                ("guidance.py", "from .runtime_identity import _PLUGIN_ROOT\n")])
def test_reviewed_root_data_flow_mutations_fail(builder, tmp_path, module, prefix):
    payload = tmp_path / "payload"
    builder.build(git(ROOT, "rev-parse", "HEAD"), payload, tmp_path / "manifest.json")
    assert_reviewed_sites(payload)
    with (payload / module).open("a") as stream:
        stream.write("\n" + prefix + '(_PLUGIN_ROOT / "prompts" / "extra.md").read_text()\n')
    with pytest.raises(AssertionError):
        assert_reviewed_sites(payload)
    print(f"M03/M04 data-flow mutation rejected: {module}")


def assert_payload_markdown_paths(payload):
    repo_dirs = {p.split("/")[0] for p in git(ROOT, "ls-files").splitlines() if "/" in p}
    checked = []
    for page in sorted(payload.rglob("*.md")):
        # Absolute URLs name external resources, not files in the installed payload.
        text = re.sub(r"\[[^\]]*\]\(https?://[^)]*\)|https?://[^\s`<>)]*", "", page.read_text())
        for match in re.finditer(r"(?<![\w/])(?:\.{1,2}/)*(?:[\w.-]+/)*[\w.-]+\.(?:md|py|sh|json|ya?ml|txt)\b", text):
            target = match.group().split("#")[0]
            if "/" not in target and not target.endswith(".md"):
                continue  # Bare host config filenames are not repository references.
            if ("/" in target and target.split("/")[0] not in repo_dirs | {".", ".."}
                    and not (page.parent / target.split("/")[0]).is_dir()):
                continue  # e.g. Hermes-owned agent/ and hermes_cli/ source references.
            candidates = [(page.parent / target).resolve(), (payload / target).resolve()]
            assert any(p.is_relative_to(payload.resolve()) and p.is_file() for p in candidates), (page, target)
            checked.append(target)
    return checked


def test_payload_markdown_link_integrity(builder, tmp_path):
    payload = tmp_path / "payload"
    builder.build(git(ROOT, "rev-parse", "HEAD"), payload, tmp_path / "manifest.json")
    assert "references/configuration.md" in assert_payload_markdown_paths(payload)  # Positive control.
    page = payload / "skills/hermes-lcm/references/configuration.md"
    with page.open("a") as stream:
        stream.write("\nSee `docs/operator-guide.md`.\n")
    with pytest.raises(AssertionError, match="docs/operator-guide"):
        assert_payload_markdown_paths(payload)
    print("relative docs mutation rejected; shipped reference positive control passed")


def test_scanner_skip_guard_requires_both_host_variables(monkeypatch):
    import os

    tree = ast.parse((ROOT / "tests/test_catalog_scanner_on.py").read_text())
    mark = next(n.value for n in tree.body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "pytestmark" for t in n.targets))
    for python, src in [("", ""), ("python", ""), ("", "src"), ("python", "src")]:
        monkeypatch.setenv("LCM_REAL_HERMES_PYTHON", python)
        monkeypatch.setenv("LCM_REAL_HERMES_SRC", src)
        skip = eval(compile(ast.Expression(mark), "skip guard", "eval"), dict(pytest=pytest, os=os))
        assert skip.args[0] == (not (python and src))


@pytest.mark.parametrize("staged", [False, True])
def test_dirty_checkout_refused(builder, source, tmp_path, staged):
    (source / "core.py").write_text("CHANGED = True\n")
    if staged:
        git(source, "add", "core.py")
    with pytest.raises(ValueError, match="dirty"):
        build(builder, source, tmp_path)
    assert not (tmp_path / "payload").exists()


def test_untracked_receipt_does_not_change_payload(builder, source, tmp_path):
    (source / "receipt.json").write_text("{}")
    result = build(builder, source, tmp_path)
    assert "receipt.json" not in {row["path"] for row in result["files"]}


@pytest.mark.parametrize("mutation", ["extra", "removed", "critical"])
def test_scanner_occurrence_mutations_fail(tmp_path, mutation):
    probe = ast.parse((ROOT / "tests/fixtures/catalog_scanner_probe.py").read_text())
    function = next(node for node in probe.body if isinstance(node, ast.FunctionDef) and node.name == "observe")
    rows = [("proc_access", "high", "cpu_capacity.py")] * 2 + [
        ("python_subprocess", "medium", "inactive_record.py"),
        ("python_subprocess", "medium", "runtime_identity.py")]
    namespace = dict(Counter=Counter, json=json, Path=Path, phase="candidate", events=[],
                     expected=ROOT / "tests/fixtures/catalog-scanner-findings.json",
                     guard=SimpleNamespace(PLUGIN_SCANNER_VERSION="fixture"))
    result = SimpleNamespace(verdict="caution", findings=[])
    namespace["original"] = lambda *a, **kw: result
    exec(compile(ast.Module(body=[function], type_ignores=[]), "observe", "exec"), namespace)

    def findings(values):
        result.findings = [SimpleNamespace(pattern_id=r, severity=s, file=f) for r, s, f in values]

    findings(rows)
    assert namespace["observe"]() is result
    if mutation == "extra":
        rows.append(rows[0])
    elif mutation == "removed":
        rows.pop(0)
    else:
        rows.append(("fixture_critical", "critical", "cpu_capacity.py"))
    findings(rows)
    with pytest.raises(AssertionError) as error:
        namespace["observe"]()
    print(f"scanner mutation {mutation} rejected: {error.value}")


def test_workflow_ref_and_queue_contract():
    workflow = (ROOT / ".github/workflows/real-host.yml").read_text()
    original_checkout = workflow.split("      - name: Set up Python (harness)")[0]
    assert "ref:" not in original_checkout
    assert "path: scanner-source" in workflow
    regressions = workflow.index("      - name: Run real-host regressions")
    scanner = workflow.index("      - name: catalog-scanner-on")
    assert regressions < scanner  # Scanner failure cannot suppress the legacy suite.
    publication = (ROOT / ".github/workflows/plugin-payload.yml").read_text()
    assert "  queue: max\n" in publication
    assert "  cancel-in-progress: false\n" in publication


def test_release_order_and_idempotence(source, tmp_path):
    import os
    import shutil

    # Exercise the actual workflow shell against local Git and a stub gh, never GitHub.
    (source / "scripts").mkdir()
    shutil.copyfile(ROOT / "scripts/build_plugin_payload.py", source / "scripts/build_plugin_payload.py")
    (source / "release").mkdir()
    (source / "release/payload-assets.txt").write_text("plugin.yaml\nLICENSE\nskills/hermes-lcm/**\n")
    commit(source)
    git(source, "tag", "v1")
    origin = tmp_path / "origin.git"
    git(tmp_path, "init", "--bare", str(origin))
    git(source, "remote", "add", "origin", str(origin))
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    gh = bin_dir / "gh"
    gh.write_text(f"#!{shutil.which('python3')}\nimport os, pathlib, sys\n"
                  "with open(os.environ['GH_LOG'], 'a') as log: log.write(repr(sys.argv[1:])+'\\n')\n"
                  "if sys.argv[1:3] == ['release','view']:\n"
                  " p=pathlib.Path(os.environ['GH_VIEW_COUNT']); n=int(p.read_text()) if p.exists() else 0\n"
                  " p.write_text(str(n+1))\n"
                  " if n < int(os.environ.get('GH_FAIL_VIEW', '0')): sys.exit(1)\n"
                  "if sys.argv[1:3] == ['release','upload']:\n"
                  " pathlib.Path(os.environ['GH_UPLOAD']).write_bytes(pathlib.Path(sys.argv[4]).read_bytes())\n")
    gh.chmod(0o755)
    sleeper = bin_dir / "sleep"
    sleeper.write_text("#!/bin/sh\nprintf '%s\\n' \"$*\" >> \"$SLEEP_LOG\"\n")
    sleeper.chmod(0o755)
    git_wrapper = bin_dir / "git"
    git_wrapper.write_text(f"#!{shutil.which('python3')}\nimport os, sys\n"
                           "if sys.argv[1] == 'push':\n"
                           " with open(os.environ['PUSH_LOG'], 'a') as log: log.write('push\\n')\n"
                           f"os.execv({shutil.which('git')!r}, [{shutil.which('git')!r}, *sys.argv[1:]])\n")
    git_wrapper.chmod(0o755)
    shell = textwrap.dedent((ROOT / ".github/workflows/plugin-payload.yml").read_text().split("        run: |\n")[1])
    base_env = dict(os.environ, PATH=f"{bin_dir}:{Path(sys.executable).parent}:" + os.environ["PATH"], SOURCE_TAG="v1",
                    PUSH_LOG=str(tmp_path / "push.log"), GH_LOG=str(tmp_path / "gh.log"), GH_UPLOAD=str(tmp_path / "uploaded.json"),
                    GITHUB_REPOSITORY="fixture/repo")

    def publish(label, failures=0, tag="v1"):
        scratch = tmp_path / label
        scratch.mkdir()
        env = dict(base_env, RUNNER_TEMP=str(scratch), SOURCE_TAG=tag, GH_FAIL_VIEW=str(failures),
                   GH_VIEW_COUNT=str(scratch / "views"), SLEEP_LOG=str(scratch / "sleeps"))
        completed = subprocess.run(["bash", "-e", "-c", shell], cwd=source, env=env,
                                   capture_output=True, text=True, timeout=60)
        print(f"publication {label}: exit={completed.returncode}\n{completed.stdout}{completed.stderr}")
        return completed

    missing = publish("missing", 1000)
    assert missing.returncode == 1
    assert "STOP: no GitHub release for v1 after 10 min; nothing pushed; re-run with workflow_dispatch once the release exists" in missing.stdout
    assert (tmp_path / "missing/sleeps").read_text().splitlines() == ["10"] * 60
    assert not (tmp_path / "missing/plugin").exists()
    assert not (tmp_path / "uploaded.json").exists()
    assert not (tmp_path / "push.log").exists()
    refs = git(origin, "for-each-ref", "refs/heads/dist")
    assert refs == "", missing.stdout + missing.stderr
    first = publish("first", 3)
    assert first.returncode == 0, first.stdout + first.stderr
    assert (tmp_path / "first/views").read_text() == "4"
    assert (tmp_path / "first/sleeps").read_text().splitlines() == ["10"] * 3
    gh_calls = (tmp_path / "gh.log").read_text().splitlines()
    assert sum("'upload'" in line for line in gh_calls) == 1
    assert (tmp_path / "push.log").read_text().splitlines() == ["push"]
    dist = git(origin, "rev-parse", "dist")
    assert git(origin, "rev-list", "--count", "dist") == "1"
    stored = git(origin, "show", dist + ":payload-manifest.json")
    assert json.loads(stored)["tooling_sha"] == git(source, "rev-parse", "HEAD")
    # A later tooling checkout must reuse the old manifest, not rebuild provenance.
    commit(source)
    second = publish("second")
    assert second.returncode == 0, second.stdout + second.stderr
    assert git(origin, "rev-parse", "dist") == dist
    assert json.loads((tmp_path / "uploaded.json").read_text()) == json.loads(stored)
    assert git(origin, "rev-list", "--count", "dist") == "1"
    git(source, "tag", "v2")
    next_tag = publish("next", tag="v2")
    assert next_tag.returncode == 0, next_tag.stdout + next_tag.stderr
    tip = git(origin, "rev-parse", "dist")
    assert git(origin, "rev-parse", "dist^") == dist
    older = publish("older")
    assert older.returncode == 0, older.stdout + older.stderr
    assert git(origin, "rev-parse", "dist") == tip
    assert json.loads((tmp_path / "uploaded.json").read_text()) == json.loads(stored)
