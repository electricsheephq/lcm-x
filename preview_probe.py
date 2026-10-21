"""Record the existing #1718 corpus plus generated wrappers, without retaining bulk inputs."""
import hashlib
import json
import os
from pathlib import Path

ROWS = {}


def pytest_configure(config):
    import hermes_lcm.externalize as ext

    original = ext._media_payload_preview_content

    def record(content):
        result = original(content)
        ROWS[hashlib.sha256(content.encode()).hexdigest()] = {
            "length": len(content), "preview": result,
        }
        return result

    ext._media_payload_preview_content = record
    caption = "deployment failed on the zebracluster"
    for key in ("caption", "alt", "title", "text", "content", "description", "name"):
        for kind in (None, "image", "deployment_error", "text"):
            part = {"image": b"\x89PNG\r\n", key: caption}
            if kind is not None:
                part["type"] = kind
            for wrapper in (lambda x: x, lambda x: [x], lambda x: (x,)):
                text = repr(wrapper(part))
                for prefix in ("", "[] ", "{} "):
                    record(prefix + text)


def pytest_unconfigure(config):
    Path(os.environ["PREVIEW_PROBE_OUT"]).write_text(json.dumps(ROWS, indent=2, sort_keys=True) + "\n")
