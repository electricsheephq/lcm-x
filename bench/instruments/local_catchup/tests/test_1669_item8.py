"""Issue #1669: round-five instrument claims, reproduced before implementation."""
from pathlib import Path

import pytest

from bench.instruments.local_catchup import floor

pytest_plugins = ["bench.instruments.local_catchup.tests.test_corpus_audit"]
WORKFLOW = Path(__file__).resolve().parents[4]/".github/workflows/lcmx-1586-floor.yml"


@pytest.mark.parametrize("swap", ["max", "1073741824", "0"])
def test_item8_effective_swap_limit_required_and_recorded(tmp_path, monkeypatch, swap):
    group = tmp_path/"cgroup"
    group.mkdir()
    (group/"memory.max").write_text("4294967296")
    (group/"memory.swap.max").write_text(swap)
    read = Path.read_text
    def read_text(path, *a, **k):
        if str(path) == "/proc/self/cgroup":
            return "0::/test\n"
        if str(path) == "/proc/self/status":
            return "Cpus_allowed_list:\t0-1\n"
        if str(path).startswith("/sys/fs/cgroup/test/"):
            return read(group/path.name)
        return read(path, *a, **k)
    resolve = Path.resolve
    monkeypatch.setattr(Path, "read_text", read_text)
    monkeypatch.setattr(Path, "resolve", lambda p: group if str(p) == "/sys/fs/cgroup/test" else resolve(p))
    if swap != "0":
        with pytest.raises(RuntimeError, match="limits"):
            floor.measured_limits(group)
    else:
        assert floor.measured_limits(group)["memory_swap_max_bytes"] == 0
    assert "memory.swap.max" in WORKFLOW.read_text()
