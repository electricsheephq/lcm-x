"""Offline regressions for per-run Codex home isolation."""
import ast
import hashlib
import json
import os
import re
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "bench/instruments/compaction_probe/track_s/s4/run_s_codex.py"


def functions(*names, **namespace):
    namespace.setdefault("Path", Path)
    tree = ast.parse(SCRIPT.read_text())
    tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    exec(compile(tree, str(SCRIPT), "exec"), namespace)
    return namespace


def test_run_home_isolation(tmp_path):
    s4 = tmp_path / "s4"
    run_home = functions("run_home", S4=s4, re=re)["run_home"]
    assert run_home("d1-r1") == s4 / "home" / "d1-r1"
    assert run_home("d1-r2") == s4 / "home" / "d1-r2"
    assert run_home("d1-r1") != run_home("d1-r2")
    assert run_home("A._-09") == s4 / "home" / "A._-09"
    assert run_home("a" * 64).name == "a" * 64


@pytest.mark.parametrize("run", ["../x", "", "a/b", ".", "..", "a" * 65, "a\n", "é"])
def test_run_home_rejects_invalid_ids(tmp_path, run):
    run_home = functions("run_home", S4=tmp_path / "s4", re=re)["run_home"]
    with pytest.raises(SystemExit, match="^STOP:"):
        run_home(run)
    assert not (tmp_path / "s4").exists()


@pytest.mark.parametrize("refresh_before_b", [True, False])
def test_parallel_setup_preserves_refresh(tmp_path, refresh_before_b):
    source = tmp_path / "synthetic.json"
    source.write_text('{"last_refresh": "synthetic-original"}')
    ns = functions("run_home", "setup_home", "sha", S4=tmp_path / "s4",
        USER_AUTH=source, os=os, re=re, shutil=shutil, hashlib=hashlib, json=json,
        codex_bin=lambda: "fake", env=lambda: {}, subprocess=SimpleNamespace(
            DEVNULL=-3, run=lambda *a, **k: SimpleNamespace(
                returncode=0, stdout="logged in chatgpt", stderr="")))
    # Use the original binding on the base so the race fails on its lost refresh.
    tree = ast.parse(SCRIPT.read_text())
    binding = next(n for n in tree.body if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == "HOME" for t in n.targets))
    exec(compile(ast.Module(body=[binding], type_ignores=[]), str(SCRIPT), "exec"), ns)
    home_for = ns.get("run_home", lambda run: ns["HOME"])
    home_a, home_b = home_for("d1-r1"), home_for("d1-r2")
    ns["HOME"] = home_a
    initial = ns["setup_home"]()
    refreshed = b'{"last_refresh": "synthetic-refreshed"}'
    if refresh_before_b:
        (home_a / "auth.json").write_bytes(refreshed)
    ns["HOME"] = home_b
    ns["setup_home"]()
    if not refresh_before_b:
        (home_a / "auth.json").write_bytes(refreshed)
    assert ns["sha"](home_a / "auth.json")[:12] != initial["auth_copy_sha256_prefix"]
    assert (home_b / "auth.json").read_bytes() == source.read_bytes()
    assert home_a.parent.stat().st_mode & 0o777 == 0o700
    assert home_a.stat().st_mode & 0o777 == 0o700
    assert home_b.stat().st_mode & 0o777 == 0o700


def test_main_rebinds_home_before_setup():
    tree = ast.parse(SCRIPT.read_text())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    parsed = next(i for i, n in enumerate(main.body) if ast.unparse(n) == "a = ap.parse_args()")
    assert ast.unparse(main.body[parsed + 1]) == "global HOME"
    assert ast.unparse(main.body[parsed + 2]) == "HOME = run_home(a.run)"
    setup = next(n for n in ast.walk(main) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Name) and n.func.id == "setup_home")
    assert main.body[parsed + 2].lineno < setup.lineno
