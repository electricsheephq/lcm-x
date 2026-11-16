"""Recreate the former large fixture at test time; no kill/recovery claim is made."""
import json
from pathlib import Path
import sys


def generate(path):
    with path.open("w", encoding="utf-8") as handle:
        for i in range(-1, 1100):
            uid = "bootstrap" if i == -1 else f"kill-{i}"
            content = "bootstrap" if i == -1 else "synthetic batch " + "z" * 4096
            entry = dict(type="user", uuid=uid, sessionId="killed", timestamp="2000-01-01T00:00:00Z",
                         message=dict(role="user", content=content))
            handle.write(json.dumps(entry) + "\n")


if __name__ == "__main__":
    generate(Path(sys.argv[1]))
