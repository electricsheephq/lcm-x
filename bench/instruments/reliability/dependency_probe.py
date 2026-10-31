"""Host-interpreter half of #1186 qualification; synthetic homes only.

No dependency preparation or model construction occurs in this probe. Mutating
admission is exercised exclusively by the harness through the native CLI.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


def state() -> dict:
    from pm.environments import (
        committed_venv, install_state_dir, project_python, runtime_facts_path, selected_venv,
    )
    from pm.paths import repo_root
    root = repo_root()
    facts = runtime_facts_path(root)
    fact = json.loads(facts.read_text()).get("packages", {}).get("venv", {}) if facts.exists() else {}
    return {
        "stamp": fact.get("stamp"), "environment": str(selected_venv(root)),
        "state_dir": str(install_state_dir(root)),
        "python": str(project_python(root)), "committed": committed_venv(root) is not None,
    }


def main() -> None:
    mode = sys.argv[1]
    if mode == "scan":
        from hermes_cli.plugins_cmd import _scan_on_install_enabled
        from tools.plugin_guard import scan_plugin
        result = scan_plugin(Path(sys.argv[2]), source="qualification")
        print(json.dumps({"scanner_enabled": _scan_on_install_enabled(),
                          "verdict": result.verdict,
                          "findings": [{"file": f.file, "rule": f.pattern_id}
                                       for f in result.findings]}))
        return
    if mode == "locked-tools":
        import hashlib
        import os
        from pm.install import ensure
        from pm.paths import store_root
        from pm.store import Store
        assert os.environ.get("LCM_1186_HOSTED") == "1", "hosted CI only"
        expected = {"python": "0ab3305457051cd3e7c031857e005f1bda17c218a1990567dacaaac6dd1d14f0",
                    "uv": "600cf9a742aca00d292673b16b5acffaa7b8c269a364ad0c2e79498dcb1fe101"}
        lock = json.loads((Path(__import__("pm").__file__).parent / "lock.json").read_text())
        for name, digest in expected.items():
            assert lock["packages"][name]["artifacts"]["linux-x64"]["sha256"] == digest
        verified = []
        store = Store(store_root())
        for name, digest in expected.items():
            artifact = lock["packages"][name]["artifacts"]["linux-x64"]
            with store.scratch() as scratch:
                archive = store.fetch(artifact["url"], digest, scratch)
                assert hashlib.sha256(archive.read_bytes()).hexdigest() == digest
            # Host ensure consumes the verified fetch entry and removes its
            # archive after publication; hash before use, not after cleanup.
            ensure(name, explicit=True)
            verified.append({"tool": name, "sha256": digest, "verified": True})
        print(json.dumps({"archives": verified}))
        return
    if mode == "preflight":
        from pm._uv import _toolchain
        from pm.runtime import _resident_runtime
        print(json.dumps({"locked_tools_ready": _resident_runtime() is not None
                          or _toolchain(realize=False) is not None}))
        return
    if mode == "state":
        print(json.dumps(state()))
        return
    if mode == "lease":
        from hermes_cli.runtime_state import lease_generation
        release = lease_generation(Path(state()["environment"]))
        print("LEASE_READY", flush=True)
        try:
            sys.stdin.read()
        finally:
            release()
        return
    if mode == "leases":
        from hermes_cli.runtime_state import leases_held
        root = Path(state()["state_dir"]) / "environments"
        print(json.dumps({p.name: leases_held(p) for p in root.iterdir() if p.is_dir()}))
        return

    # The harness chooses the committed PM generation's interpreter after
    # admission. Bootstrap also applies the host's normal activation/path rules.
    if len(sys.argv) > 2:
        assert Path(sys.prefix).resolve() == Path(sys.argv[2]).resolve(), (
            "probe did not start in the selected PM generation", sys.prefix, sys.argv[2])
    import hermes_bootstrap  # noqa: F401
    from hermes_cli import plugins as plugins
    plugins.discover_plugins(force=True)
    manager = plugins.get_plugin_manager()
    core = manager._plugins.get("hermes-lcm-x")
    companion = manager._plugins.get("hermes-lcm-x-local-embeddings")
    result = {
        "runtime_prefix": sys.prefix, "runtime_executable": sys.executable,
        "core_enabled": bool(core and core.enabled), "core_error": getattr(core, "error", None),
        "companion_enabled": bool(companion and companion.enabled),
        "companion_error": getattr(companion, "error", None),
        "fastembed_spec": importlib.util.find_spec("fastembed") is not None,
    }
    if mode == "import":
        import fastembed
        result["fastembed_imported"] = bool(fastembed)
    elif mode == "fts":
        assert not result["fastembed_spec"], "decline home must start without fastembed"
        engine = manager._context_engine
        assert engine is not None, result
        token = "quartzneedle1186"
        history = [{"role": "user", "content": f"The synthetic codeword is {token}."},
                   {"role": "assistant", "content": f"Recorded {token}."}]
        plugins.invoke_hook(
            "post_llm_call", session_id="dependency-fixture", task_id="fixture", turn_id="one",
            user_message=history[0]["content"], assistant_response=history[1]["content"],
            conversation_history=history, model="synthetic", platform="cli",
        )
        result["recall"] = json.loads(engine.handle_tool_call(
            "lcm_recall", {"query": token, "include": "verbatim", "limit": 5}))
        result["doctor"] = json.loads(engine.handle_tool_call("lcm_doctor", {}))
        assert token in json.dumps(result["recall"]), result
        assert result["recall"].get("degraded") is True, result
        assert f"HERMES_HOME={engine._hermes_home} hermes -p default plugins enable hermes-lcm-x-local-embeddings" in json.dumps(result["doctor"]), result
    else:
        raise ValueError(mode)
    assert result["core_enabled"] and not result["core_error"], result
    print(json.dumps(result))


if __name__ == "__main__":
    main()
