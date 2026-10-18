"""Synthetic-probe-only pytest plugin: immutable built inputs and SQL snapshots."""
from __future__ import annotations

from collections.abc import Callable, Iterator
from copy import deepcopy
from functools import wraps
import hashlib
from importlib import import_module
import json
import os
from pathlib import Path
from typing import Any

import pytest


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def snapshot(obj: Any) -> list[list[Any]]:
    store = getattr(obj, "_store", None)
    if store is None:
        return []
    try:
        return [list(row) for row in store.connection.execute(
            "SELECT store_id, session_id, role, content, observed_at "
            "FROM messages ORDER BY store_id")]
    except Exception:
        # Closed stores are recorded as unavailable, never fabricated empty proof.
        return []


@pytest.fixture(autouse=True)
def capture_built_inputs(monkeypatch: Any, request: Any) -> Iterator[None]:
    destination = os.environ.get("D1525_CAPTURE_DIR")
    if not destination:
        pytest.fail("D1525_CAPTURE_DIR is required by capture_inputs")
    root = Path(os.environ["D1525_PROBES_ROOT"]).resolve()
    engine = import_module("hermes_lcm.engine").LCMEngine
    events: list[dict[str, Any]] = []
    receipts: list[str] = []
    depth = 0
    groups: dict[str, int] = {}
    identities: dict[str, list[str]] = {}
    write_text = Path.write_text

    def observe_write(path: Path, data: str, *args: Any, **kwargs: Any) -> int:
        result = write_text(path, data, *args, **kwargs)
        if path.suffix == ".json" and path.resolve().is_relative_to(root):
            receipts.append(path.resolve().relative_to(root).as_posix())
        return result

    monkeypatch.setattr(Path, "write_text", observe_write)

    def wrap(method: Callable[..., Any], operation: str) -> Callable[..., Any]:
        @wraps(method)
        def observed(obj: Any, *args: Any, **kwargs: Any) -> Any:
            nonlocal depth
            if depth:
                return method(obj, *args, **kwargs)
            messages = next((arg for arg in args if isinstance(arg, list)),
                            kwargs.get("messages", []))
            built = json.loads(canonical(deepcopy(messages)))
            database = str(getattr(getattr(obj, "_store", None), "db_path", ""))
            group = groups.setdefault(database, len(groups))
            event: dict[str, Any] = {
                "group": group,
                "operation": operation, "session": str(getattr(obj, "_session_id", "") or ""),
                "messages": built, "kwargs": deepcopy(kwargs), "before": snapshot(obj),
                "lifecycle_args": [arg for arg in args if isinstance(arg, (str, int, bool))],
            }
            # Reserve order before entering calls; nested internal ingests are not host views.
            events.append(event)
            depth += 1
            try:
                result = method(obj, *args, **kwargs)
                if operation == "compress" and isinstance(result, list):
                    event["returned_messages"] = json.loads(canonical(deepcopy(result)))
                return result
            except Exception as exc:
                event["error_type"] = type(exc).__name__
                raise
            finally:
                depth -= 1
                event["after"] = snapshot(obj) if operation != "shutdown" else event["before"]
                event["after_session"] = str(getattr(obj, "_session_id", "") or "")
                normalized = []
                provenance = []
                for message in built:
                    if not isinstance(message, dict):
                        continue
                    observed_identity = identities.get(canonical(message))
                    if observed_identity is not None:
                        normalized.append(observed_identity)
                        provenance.append("observed-original-call")
                    else:
                        content = message.get("content")
                        normalized.append([str(message.get("role") or "unknown"),
                                           content if isinstance(content, str) else
                                           json.dumps(content, ensure_ascii=False, sort_keys=True) if content else ""])
                        provenance.append("raw-content-fallback")
                event["normalized"] = normalized
                event["normalization_sources"] = provenance
        return observed

    original_identity = engine._message_replay_identity

    @wraps(original_identity)
    def observe_identity(obj: Any, message: Any, *args: Any, **kwargs: Any) -> Any:
        result = original_identity(obj, message, *args, **kwargs)
        if not kwargs.get("stored_row", False):
            identities[canonical(message)] = [result[0], result[1]]
        return result

    monkeypatch.setattr(engine, "_message_replay_identity", observe_identity)
    for name in ("ingest", "compress", "on_session_start", "on_session_end", "shutdown"):
        monkeypatch.setattr(engine, name, wrap(getattr(engine, name), name))
    yield
    dest = Path(destination)
    dest.mkdir(parents=True, exist_ok=True)
    node = str(request.node.nodeid)
    key = hashlib.sha256(node.encode()).hexdigest()
    data = {"plugin_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "capture_version": 3, "node": node, "receipts": sorted(set(receipts)), "events": events}
    with (dest / (key + ".json")).open("x") as stream:
        stream.write(canonical(data) + "\n")
