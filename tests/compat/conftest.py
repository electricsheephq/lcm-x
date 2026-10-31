import io
import os
from pathlib import Path
import subprocess
import tarfile

import pytest

from tests.compat import codec_adapter
from tests.compat.helpers import READERS


def missing_prerequisite(reason):
    if os.environ.get("CI"):
        pytest.fail(f"{reason}; CI must provide it")
    pytest.skip(reason)


@pytest.fixture(scope="session")
def host_root():
    try:
        import agent.context_engine as host
    except ImportError:
        missing_prerequisite("the agent.context_engine host stub is not importable")
    return Path(host.__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def old_tree(tmp_path_factory, host_root):
    trees = {}

    def extract(reader, pins=None):
        commit = (READERS if pins is None else pins)[reader]
        if commit not in trees:
            root = Path(__file__).resolve().parents[2]
            if subprocess.run(["git", "-C", str(root), "cat-file", "-e", f"{commit}^{{commit}}"],
                              capture_output=True).returncode:
                missing_prerequisite(f"the {reader} commit {commit} is not available")
            archive = subprocess.run(["git", "-C", str(root), "archive", commit],
                                     check=True, capture_output=True).stdout
            old = tmp_path_factory.mktemp(reader)
            with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
                tar.extractall(old, filter="data")
            trees[commit] = old
        return trees[commit]

    return extract


@pytest.fixture(autouse=True)
def deterministic_runtime(monkeypatch, host_root):
    for key in os.environ:
        if key.startswith("LCM_"):
            monkeypatch.delenv(key)
    codec_adapter.deterministic_runtime(monkeypatch)
    yield
    codec_adapter.clear_token_cache()
