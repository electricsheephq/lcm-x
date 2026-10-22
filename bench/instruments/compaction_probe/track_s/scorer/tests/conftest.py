"""Keep pytest scratch in the caller-selected temporary directory and expose scorer modules."""
import os
import tempfile
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
sys.dont_write_bytecode = True


def pytest_configure(config):
    base = Path(tempfile.mkdtemp(prefix="track-s-pytest-", dir=os.environ["TMPDIR"]))
    config.option.basetemp = str(base)
