"""Fault-inject host commands while real Bash runs seeds and the EXIT trap."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

DECISION = Path(__file__).resolve().parents[1]
HOST_KEYS = {"elapsed_s", "load1m_before", "load1m_after", "host_load_peak1m",
             "steal_pct", "iowait_pct", "busy_pct"}


@pytest.mark.parametrize("bash", ["bash", "/bin/bash"])
@pytest.mark.parametrize("status", [0, 3])
@pytest.mark.parametrize("failure", ["before", "after", "delta", "none", "rewrite-write", "rewrite-replace"])
def test_host_failure_preserves_seed_flow_and_cell_status(tmp_path, bash, status, failure):
    # Isolate the host failure from unrelated git preflights; existing git fixtures stay real.
    bin_dir, kit, out = tmp_path / "bin", tmp_path / "kit", tmp_path / "out"
    bin_dir.mkdir()
    (kit / "s2").mkdir(parents=True)
    (kit / "decision").mkdir()
    git = bin_dir / "git"
    git.write_text("#!/bin/sh\ncase \"$3\" in rev-parse) echo fixture;; status) :;; *) exit 99;; esac\n")
    git.chmod(0o755)
    wrapper = bin_dir / "python-wrapper"
    wrapper.write_text(f"#!{sys.executable}\n" + '''import io, os, sys
from pathlib import Path
args = sys.argv[1:]
if len(args) > 2 and Path(args[1]).name == 'hoststat.py':
    mode = args[2]
    marker = Path(os.environ['OUT']) / 'snapshot-started'
    stage = 'delta' if mode == 'delta' else 'after' if marker.exists() else 'before'
    if mode == 'snapshot': marker.touch()
    if stage == os.environ['HOST_FAILURE']: sys.exit(91)
if len(args) == 5 and args[1] == '-' and os.environ['HOST_FAILURE'].startswith('rewrite-'):
    receipt = Path(args[2])
    receipt.with_name('receipt-before.json').write_bytes(receipt.read_bytes())
    def fail_replace(*args):
        raise OSError('injected receipt replace failure')
    if os.environ['HOST_FAILURE'] == 'rewrite-replace':
        os.replace = fail_replace
    else:
        original_open = io.open
        class FailingWrite:
            def __init__(self, stream): self.stream = stream
            def __getattr__(self, name): return getattr(self.stream, name)
            def __enter__(self): return self
            def __exit__(self, *args): return self.stream.__exit__(*args)
            def write(self, text):
                self.stream.write(text[:11])
                self.stream.flush()
                raise OSError('injected receipt write failure')
        def failing_open(file, mode='r', *args, **kwargs):
            stream = original_open(file, mode, *args, **kwargs)
            return FailingWrite(stream) if mode == 'w' else stream
        io.open = failing_open
    sys.argv = ['-', *args[2:]]
    exec(compile(sys.stdin.read(), '<record_host>', 'exec'))
    sys.exit(0)
os.execv(sys.executable, [sys.executable, *args])
''')
    wrapper.chmod(0o755)
    (kit / "s2/run_s_lcmx.py").write_text('''import os, sys
from pathlib import Path
seed = sys.argv[sys.argv.index('--seed') + 1]
(Path(os.environ['TRACK_S_OUT']) / ('seed-' + seed)).touch()
''' + ("print('HTTP 429', flush=True)\n" if status else "print('seed passed')\n"))
    for name in ("score_decision", "pmw_read"):
        (kit / f"decision/{name}.py").write_text(
            f"import os\nfrom pathlib import Path\n(Path(os.environ['OUT']) / '{name}-ran').touch()\n")
    env = dict(os.environ, PATH=str(bin_dir) + os.pathsep + os.environ['PATH'],
               PRODUCT_WT=str(tmp_path), PRODUCT_SHA="fixture", KIT_DIR=str(kit),
               MATERIAL=str(tmp_path / "material"), OUT=str(out), PY=str(wrapper),
               FACTS_FLOOR="0.772", GLM_API_KEY="synthetic-unused", SEEDS="1 2 3", PARALLEL="2",
               HOST_FAILURE=failure)
    for key in ("FAULT", "S2_FAULT_SLEEP_S", "ARM", "SUMMARY_LANE", "READER"):
        env.pop(key, None)
    result = subprocess.run([bash, str(DECISION / "run_pmw_cell.sh")], env=env,
                            capture_output=True, text=True, timeout=20)
    assert result.returncode == status, result.stdout + result.stderr
    assert (out / "seed-1").exists() and (out / "seed-2").exists()
    assert (out / "seed-3").exists() == (status == 0)
    assert (out / "STOP").exists() == (status == 3)
    assert (out / "score_decision-ran").exists() == (status == 0)
    assert (out / "pmw_read-ran").exists() == (status == 0)
    if failure.startswith("rewrite-"):
        assert (out / "cell.json").read_bytes() == (out / "receipt-before.json").read_bytes()
        assert "injected receipt" in result.stderr
        return
    host = json.loads((out / "cell.json").read_text())["host"]
    if failure != "none":
        assert {key: host[key] for key in HOST_KEYS} == dict.fromkeys(HOST_KEYS)
        assert host.get("reason") and failure in host["reason"]
    else:
        assert HOST_KEYS <= host.keys()
        assert host["elapsed_s"] > 0
        assert host["load1m_before"] is not None and host["load1m_after"] is not None
        if host["steal_pct"] is None:
            assert host.get("reason")
