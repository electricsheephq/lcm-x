"""Pin construction and refusal checks on synthetic instruments."""

import hashlib
import json
from pathlib import Path
import platform
import sys

import pytest

from bench.instruments.d5_clone import pins


SOURCE = '''import re
R = re.compile(r"^row")

def f(value):
    return g(value)

def g(value):
    return bool(R.match(value))

# outside the closure
'''


@pytest.fixture
def instrument(tmp_path):
    module = tmp_path / "instrument.py"
    module.write_text(SOURCE, encoding="utf-8")
    (tmp_path / "helper.py").write_text("VALUE = 1\n", encoding="utf-8")
    manifest = pins.build(tmp_path, [module.name], {module.name: ["f"]},
                          {module.name: ["helper.py"]},
                          {"path_basename": Path(sys.executable).name, "version": platform.python_version()})
    return tmp_path, module, tmp_path / "d5-clone-pins.json", manifest


def save(path, manifest):
    path.write_text(json.dumps(manifest), encoding="utf-8")


def test_build_closure_and_verify(instrument):
    directory, module, path, manifest = instrument
    assert pins.call_closure(module, ["f"]) == ({"f", "g"}, {"R", "re"})  # The import is pinned too.
    assert manifest["schema"] == "d5-clone-pins/1"
    assert {f["qualname"] for f in manifest["functions"]} == {"f", "g"}
    assert manifest["extra_hashes"][0]["name"] == "R"
    assert json.loads(path.read_text()) == manifest
    assert pins.verify(path, directory) == {"verdict": "PASS", "mismatches": []}


def test_helper_change_detected_even_with_updated_file_pin(instrument):
    directory, module, path, manifest = instrument
    original = pins.function_source_sha256(module, "f")
    module.write_text(SOURCE.replace("bool(R.match(value))", "R.search(value)"), encoding="utf-8")
    manifest["files"][0]["sha256"] = hashlib.sha256(module.read_bytes()).hexdigest()
    save(path, manifest)
    assert pins.function_source_sha256(module, "f") == original
    assert pins.verify(path, directory)["mismatches"] == [module.name]


def test_comment_preserves_closure_but_changes_whole_file(instrument):
    directory, module, path, manifest = instrument
    module.write_text(SOURCE + "# another comment\n", encoding="utf-8")
    assert pins.verify(path, directory)["verdict"] == "FAIL"
    manifest["files"][0]["sha256"] = hashlib.sha256(module.read_bytes()).hexdigest()
    save(path, manifest)
    assert pins.verify(path, directory)["verdict"] == "PASS"


def test_assignment_and_loaded_module_changes(instrument):
    directory, module, path, manifest = instrument
    module.write_text(SOURCE.replace('r"^row"', 'r"^changed"'), encoding="utf-8")
    manifest["files"][0]["sha256"] = hashlib.sha256(module.read_bytes()).hexdigest()
    save(path, manifest)
    assert pins.verify(path, directory)["mismatches"] == [module.name]
    module.write_text(SOURCE, encoding="utf-8")
    manifest["files"][0]["sha256"] = hashlib.sha256(module.read_bytes()).hexdigest()
    save(path, manifest)
    (directory / "helper.py").write_text("VALUE = 2\n", encoding="utf-8")
    assert pins.verify(path, directory)["mismatches"] == ["helper.py"]


def test_interpreter_and_require(instrument):
    directory, _, path, manifest = instrument
    assert pins.verify(path, directory, "other")["mismatches"] == ["interpreter"]
    with pytest.raises(SystemExit) as exc:
        pins.require(path, directory, "other")
    assert exc.value.code == 2
    manifest["interpreter"]["path_basename"] = "other-python"
    save(path, manifest)
    assert pins.verify(path, directory)["mismatches"] == ["interpreter"]


def test_exact_decorated_function_and_method(tmp_path):
    path = tmp_path / "decorated.py"
    function = "@decorate(1)\n@other\nasync def f():\n    return 'é'"
    method = "@decorate\n    def method(self):\n        return 2"
    path.write_text(function + "\n\nclass C:\n    " + method + "\n", encoding="utf-8")
    assert pins.function_source_sha256(path, "f") == hashlib.sha256(function.encode()).hexdigest()
    assert pins.function_source_sha256(path, "C.method") == hashlib.sha256(method.encode()).hexdigest()
    with pytest.raises(ValueError):
        pins.function_source_sha256(path, "missing")


def test_closure_resolves_globals_and_callback_references(tmp_path):
    module = tmp_path / "scope.py"
    module.write_text('''R = 2
S = R + 1
def g():
    return S
def f():
    callback = g
    return callback
def shadow(g):
    return g()
def nested():
    def inner():
        return g()
    return inner()
''', encoding="utf-8")
    assert pins.call_closure(module, ["f"]) == ({"f", "g"}, {"R", "S"})
    assert pins.call_closure(module, ["shadow"]) == ({"shadow"}, set())
    assert pins.call_closure(module, ["nested"]) == ({"nested", "g"}, {"R", "S"})


@pytest.mark.parametrize("damage", ["missing", "malformed", "schema", "kind", "path"])
def test_bad_manifest_fails_closed(instrument, damage):
    directory, _, path, manifest = instrument
    if damage == "missing":
        path = directory / "absent.json"
    elif damage == "malformed":
        path.write_text("{", encoding="utf-8")
    else:
        if damage == "schema":
            manifest["schema"] = "unknown"
        elif damage == "kind":
            manifest["extra_hashes"][0]["kind"] = "unknown"
        else:
            manifest["files"][0]["name"] = "../instrument.py"
        save(path, manifest)
    result = pins.verify(path, directory)
    assert result["verdict"] == "FAIL"
    assert all(".." not in name and not name.startswith("/") for name in result["mismatches"])


def test_empty_manifest_fails(instrument):
    directory, _, path, manifest = instrument
    save(path, {**manifest, "files": [], "functions": [], "extra_hashes": []})
    assert pins.verify(path, directory) == {"verdict": "FAIL", "mismatches": ["empty", "unpinned_files"]}


def _pin(tmp_path, source, roots=("rule",), loads=(), files=None):
    module = tmp_path / "rules.py"
    module.write_text(source, encoding="utf-8")
    for name, text in (files or {}).items():
        (tmp_path / name).write_text(text, encoding="utf-8")
    pins.build(tmp_path, [module.name], {module.name: list(roots)}, {module.name: list(loads)},
               {"path_basename": Path(sys.executable).name, "version": platform.python_version()})
    return module, tmp_path / "d5-clone-pins.json"


def _repin_file_only(path, module):
    """Re-pin only the whole-file sha, as an edition with a changed helper would present itself."""
    manifest = json.loads(path.read_text())
    manifest["files"][0]["sha256"] = hashlib.sha256(module.read_bytes()).hexdigest()
    save(path, manifest)


@pytest.mark.parametrize("before,after", [
    ("class Helper:\n    def run(self):\n        return 1\n", "class Helper:\n    def run(self):\n        return 2\n"),
    ("from json import dumps as h\n", "from pickle import dumps as h\n"),
    ("A, B = 1, 2\n", "A, B = 1, 3\n"),
], ids=["class-method", "import-alias", "tuple-assign"])
def test_closure_pins_classes_imports_and_tuple_targets(tmp_path, before, after):
    use = {"class-method": "Helper().run()", "import-alias": "h", "tuple-assign": "B"}
    key = next(k for k, v in {"class-method": "Helper", "import-alias": "h", "tuple-assign": "B"}.items()
               if v in before)
    rule = f"\n\ndef rule():\n    return {use[key]}\n"
    module, path = _pin(tmp_path, before + rule)
    assert pins.verify(path, tmp_path)["verdict"] == "PASS"
    module.write_text(after + rule, encoding="utf-8")
    _repin_file_only(path, module)
    assert pins.verify(path, tmp_path)["verdict"] == "FAIL"


@pytest.mark.parametrize("source", [
    "if True:\n    def h():\n        return 1\n\ndef rule():\n    return h()\n",
    "try:\n    X = 1\nexcept Exception:\n    X = 2\n\ndef rule():\n    return X\n",
    "L = [1]\nL += [2]\n\ndef rule():\n    return L\n",
    "D = {}\nD['k'] = 1\n\ndef rule():\n    return D\n",
    "def h():\n    return 1\n\ndef rule():\n    return globals()['h']()\n",
    "def h():\n    return 1\n\ndef h():\n    return 2\n\ndef rule():\n    return h()\n",
], ids=["def-in-if", "assign-in-try", "augassign", "subscript-store", "globals-lookup", "rebound-def"])
def test_closure_refuses_unpinnable_bindings(tmp_path, source):
    with pytest.raises(ValueError):
        _pin(tmp_path, source)


def test_local_import_must_be_listed_in_loads(tmp_path):
    source = "from . import sibling\n\ndef rule():\n    return sibling.VALUE\n"
    with pytest.raises(ValueError, match="loads"):
        _pin(tmp_path, source, files={"sibling.py": "VALUE = 1\n"})
    _pin(tmp_path, source, loads=["sibling.py"])


def test_version_checked_by_default_and_manifest_consistency(tmp_path):
    module, path = _pin(tmp_path, "def rule():\n    return 1\n")
    manifest = json.loads(path.read_text())
    save(path, {**manifest, "interpreter": {**manifest["interpreter"], "version": "3.0.0"}})
    assert pins.verify(path, tmp_path)["mismatches"] == ["interpreter"]
    save(path, {**manifest, "files": manifest["files"] * 2})
    assert "manifest" in pins.verify(path, tmp_path)["mismatches"]
    stray = {**manifest["functions"][0], "file": "other.py"}
    save(path, {**manifest, "functions": manifest["functions"] + [stray]})
    assert "manifest" in pins.verify(path, tmp_path)["mismatches"]


def test_decorator_with_space_after_at_is_in_the_segment(tmp_path):
    path = tmp_path / "spaced.py"
    function = "@ decorate\ndef f():\n    return 1"
    path.write_text(function + "\n", encoding="utf-8")
    assert pins.function_source_sha256(path, "f") == hashlib.sha256(function.encode()).hexdigest()


@pytest.mark.parametrize("importer,extra", [
    ("from {pkg}.sibling import VALUE\n\ndef rule():\n    return VALUE\n", {}),
    ("import sibpkg\n\ndef rule():\n    return sibpkg.VALUE\n", {"sibpkg/__init__.py": "VALUE = 1\n"}),
], ids=["package-path", "sibling-package"])
def test_absolute_and_package_imports_must_be_listed(tmp_path, importer, extra):
    directory = tmp_path / "pkgdir"
    directory.mkdir()
    (directory / "sibling.py").write_text("VALUE = 1\n", encoding="utf-8")
    for name, text in extra.items():
        (directory / name).parent.mkdir(exist_ok=True)
        (directory / name).write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="loads"):
        _pin(directory, importer.format(pkg=directory.name))


def test_second_level_import_must_be_listed(tmp_path):
    files = {"sib.py": "from sib2 import X\n", "sib2.py": "X = 1\n"}
    with pytest.raises(ValueError, match="loads"):
        _pin(tmp_path, "import sib\n\ndef rule():\n    return sib.X\n", loads=["sib.py"], files=files)
    _pin(tmp_path, "import sib\n\ndef rule():\n    return sib.X\n", loads=["sib.py", "sib2.py"])


@pytest.mark.parametrize("body", ["importlib.import_module('sib').X", "sys.modules['sib'].X"])
def test_dynamic_module_access_is_refused(tmp_path, body):
    with pytest.raises(ValueError):
        _pin(tmp_path, f"import importlib\nimport sys\n\ndef rule():\n    return {body}\n")


def test_data_names_cannot_exempt_code(tmp_path):
    module = tmp_path / "rules.py"
    module.write_text("def h():\n    return 1\n\ndef rule():\n    return h()\n", encoding="utf-8")
    with pytest.raises(ValueError, match="plain values"):
        pins.call_closure(module, ["rule"], data=["h"])


@pytest.mark.parametrize("loader", [
    "import sibpkg\n\ndef rule():\n    return sibpkg.h()\n",
    "import runpy\n\ndef rule():\n    return runpy.run_path('sibpkg/mod.py')\n",
    "from importlib import import_module as im\nMOD = im('sibpkg.mod')\n\ndef rule():\n    return MOD.h()\n",
], ids=["package-relative-submodule", "runpy", "aliased-import-module"])
def test_every_file_under_the_instrument_dir_is_pinned(tmp_path, loader):
    (tmp_path / "sibpkg").mkdir()
    (tmp_path / "sibpkg" / "__init__.py").write_text("from .mod import h\n", encoding="utf-8")
    (tmp_path / "sibpkg" / "mod.py").write_text("def h():\n    return 1\n", encoding="utf-8")
    (tmp_path / "rules.py").write_text(loader, encoding="utf-8")
    interpreter = {"path_basename": Path(sys.executable).name, "version": platform.python_version()}
    with pytest.raises(ValueError, match="every file"):  # sibpkg/mod.py is not listed.
        pins.build(tmp_path, ["rules.py"], {"rules.py": []}, {"rules.py": ["sibpkg/__init__.py"]}, interpreter)
    pins.build(tmp_path, ["rules.py"], {"rules.py": []},
               {"rules.py": ["sibpkg/__init__.py", "sibpkg/mod.py"]}, interpreter)
    path = tmp_path / "d5-clone-pins.json"
    assert pins.verify(path, tmp_path)["verdict"] == "PASS"
    (tmp_path / "sibpkg" / "mod.py").write_text("def h():\n    return 2\n", encoding="utf-8")
    assert pins.verify(path, tmp_path)["mismatches"] == ["sibpkg/mod.py"]


def test_added_file_or_symlink_fails_verify(tmp_path):
    module, path = _pin(tmp_path, "def rule():\n    return 1\n")
    (tmp_path / "late.py").write_text("X = 1\n", encoding="utf-8")
    assert pins.verify(path, tmp_path)["mismatches"] == ["unpinned_files"]
    (tmp_path / "late.py").unlink()
    (tmp_path / "link.py").symlink_to(module)
    assert pins.verify(path, tmp_path)["mismatches"] == ["unpinnable_files"]


@pytest.mark.parametrize("planted", ["__pycache__/rules.cpython-312.pyc", "stray.pyc"])
def test_bytecode_under_the_instrument_dir_fails_verify(tmp_path, planted):
    module, path = _pin(tmp_path, "def rule():\n    return 1\n")
    (tmp_path / planted).parent.mkdir(exist_ok=True)
    (tmp_path / planted).write_bytes(b"\x00planted")
    assert pins.verify(path, tmp_path)["mismatches"] == ["unpinnable_files"]


def test_a_file_named_like_a_marker_cannot_cancel_a_symlink(tmp_path):
    """A pinned ordinary file named like an old rejection marker must not hide a directory link."""
    target = tmp_path.parent / (tmp_path.name + "-outside")
    target.mkdir()
    (target / "rule.py").write_text("VALUE = 1\n", encoding="utf-8")
    (tmp_path / "symlink:pkg").write_text("decoy\n", encoding="utf-8")
    (tmp_path / "bytecode:x").write_text("decoy\n", encoding="utf-8")
    module, path = _pin(tmp_path, "def rule():\n    return 1\n", loads=["symlink:pkg", "bytecode:x"])
    assert pins.verify(path, tmp_path)["verdict"] == "PASS"
    (tmp_path / "pkg").symlink_to(target, target_is_directory=True)
    assert pins.verify(path, tmp_path)["mismatches"] == ["unpinnable_files"]
    (target / "rule.py").write_text("VALUE = 2\n", encoding="utf-8")
    assert pins.verify(path, tmp_path)["mismatches"] == ["unpinnable_files"]


def test_an_unlistable_directory_fails_verify(tmp_path):
    module, path = _pin(tmp_path, "def rule():\n    return 1\n")
    hidden = tmp_path / "h"
    hidden.mkdir()
    (hidden / "x.py").write_text("VALUE = 1\n", encoding="utf-8")
    hidden.chmod(0o311)
    try:
        assert pins.verify(path, tmp_path)["verdict"] == "FAIL"
    finally:
        hidden.chmod(0o755)


def test_a_manifest_named_file_fails_when_the_manifest_lives_elsewhere(tmp_path):
    module, path = _pin(tmp_path, "def rule():\n    return 1\n")
    outside = tmp_path.parent / (tmp_path.name + "-manifest.json")
    outside.write_bytes(path.read_bytes())
    assert pins.verify(outside, tmp_path)["mismatches"] == ["unpinned_files"]


def test_an_empty_bytecode_cache_dir_fails_verify(tmp_path):
    module, path = _pin(tmp_path, "def rule():\n    return 1\n")
    (tmp_path / "__pycache__").mkdir()
    assert pins.verify(path, tmp_path)["mismatches"] == ["unpinnable_files"]
