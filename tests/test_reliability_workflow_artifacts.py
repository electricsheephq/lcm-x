"""Keep hosted reliability cells usable by the standalone re-scorer."""

from pathlib import Path
import re
import shlex


WORKFLOW = Path(__file__).resolve().parents[1] / ".github/workflows/reliability-nightly.yml"


def test_every_matrix_run_keeps_nonpassing_cell_databases():
    workflow = WORKFLOW.read_text(encoding="utf-8")
    commands = [line for line in workflow.splitlines() if "run_matrix.py" in line]
    assert len(commands) == 3
    for command in commands:
        args = shlex.split(command)
        assert "--keep-dbs" in args
        assert args[args.index("--keep-dbs") + 1] == "fail"
        assert "--keep-homes" not in args


def test_upload_keeps_whole_cells_without_caches_or_tool_files():
    workflow = WORKFLOW.read_text(encoding="utf-8")
    upload = workflow.split("      - name: Upload results\n", 1)[1]
    paths_block = upload.split("          path: |\n", 1)[1].split("          if-no-files-found:", 1)[0]
    paths = {line.strip() for line in paths_block.splitlines() if line.strip()}
    assert "reliability-out/*/cells/**" in paths
    assert "reliability-out/default-window-*/**" in paths
    assert "!reliability-out/**/pycache/**" in paths
    assert "!reliability-out/*/cells/*/*/*/files/**" in paths
    # A file allowlist silently drops future scorer inputs. Payload-only Hermes
    # home wrappers must remain: release_scratch keeps externalized rows there.
    assert not [path for path in paths if re.search(r"/cells/.*\.[a-z]+$", path)]
    assert not any("hermes-home" in path for path in paths)
