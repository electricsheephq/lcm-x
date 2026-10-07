"""Per-file admission for Track S decision scores and loss classifications."""
import hashlib
import json
import os
import tempfile

SCHEMA = "track-s-score-manifest-v2"


def load(path):
    if not path.exists():
        return None
    manifest = json.loads(path.read_text())
    return manifest if manifest.get("schema") == SCHEMA else {"schema": SCHEMA, "entries": {}}


def write_atomic(path, manifest):
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as tmp:
        tmp.write(json.dumps(manifest, indent=1) + "\n")
    os.replace(tmp.name, path)


def admitted(manifest_dir, manifest, file_path):
    if manifest is None or not file_path.is_file():
        return False
    key = file_path.relative_to(manifest_dir).as_posix()
    entry = manifest["entries"].get(key)
    return entry is not None and hashlib.sha256(file_path.read_bytes()).hexdigest() == entry["sha256"]
