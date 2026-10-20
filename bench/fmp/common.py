"""Small shared IO helpers. Never put exception messages or corpus in logs."""
import hashlib
import json
import logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def sha(data):
    return hashlib.sha256(data).hexdigest()


def digest_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def utc():
    return datetime.now(timezone.utc).isoformat()


def commit(repo):
    # A dirty tree is not the named commit: hash its content so different states cannot share stores.
    head = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain"], text=True).strip()
    if not dirty:
        return head
    h = hashlib.sha256(subprocess.check_output(
        ["git", "-C", str(repo), "diff", "--no-ext-diff", "--binary", "HEAD"]))
    names = subprocess.check_output(["git", "-C", str(repo), "ls-files", "--others", "--exclude-standard", "-z"])
    for name in sorted(n for n in names.split(b"\0") if n):
        path = Path(repo) / name.decode()
        content = str(path.readlink()).encode() if path.is_symlink() else digest_file(path).encode()
        h.update(name + b"\0" + content + b"\0")
    return head + "-" + h.hexdigest() + "-dirty"


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    temp.replace(path)


def quiet():
    logging.disable(logging.CRITICAL)


def load_lcm(repo):
    import importlib
    import sys
    repo = Path(repo).resolve()
    for name in ("benchmarking", "hermes_lcm"):
        module = sys.modules.get(name)
        if module and not str(getattr(module, "__file__", "") or getattr(module, "__path__", [""])[0]).startswith(str(repo) + "/"):
            raise RuntimeError("LCM checkout already bound to another repository")
    sys.path.insert(0, str(repo))
    # The repository's helper bootstraps the supported Hermes host contract.
    module = importlib.import_module("benchmarking.longmemeval")
    if not Path(module.__file__).resolve().is_relative_to(repo):
        raise RuntimeError("LCM loader bound to another checkout")
    module._ensure_hermes_lcm_package()
    return module
