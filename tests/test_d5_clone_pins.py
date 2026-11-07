"""Pin construction and refusal checks on synthetic instruments."""

import hashlib
import json
from pathlib import Path
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
                          {"path_basename": Path(sys.executable).name, "version": "3.test"})
    return tmp_path, module, tmp_path / "d5-clone-pins.json", manifest


def save(path, manifest):
    path.write_text(json.dumps(manifest), encoding="utf-8")


def test_build_closure_and_verify(instrument):
    directory, module, path, manifest = instrument
    assert pins.call_closure(module, ["f"]) == ({"f", "g"}, {"R"})
    assert manifest["schema"] == "d5-clone-pins/1"
    assert {f["qualname"] for f in manifest["functions"]} == {"f", "g"}
    assert manifest["extra_hashes"][0]["name"] == "R"
    assert json.loads(path.read_text()) == manifest
    assert pins.verify(path, directory, "3.test") == {"verdict": "PASS", "mismatches": []}


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
    assert all("/" not in name for name in result["mismatches"])


def test_empty_manifest_fails(instrument):
    directory, _, path, manifest = instrument
    save(path, {**manifest, "files": [], "functions": [], "extra_hashes": []})
    assert pins.verify(path, directory, "3.test") == {"verdict": "FAIL", "mismatches": ["empty"]}
