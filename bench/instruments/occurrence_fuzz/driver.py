"""Run generated host views against one isolated checkout and emit final ledgers."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import logging
import shutil
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any


class _Catch(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.WARNING)
        self.bad: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        message = record.getMessage()
        if "ingest failed" in message or record.levelno >= logging.ERROR:
            self.bad.append(message[:80])


def load_tree(tree: Path) -> tuple[Any, Any]:
    sys.path.insert(0, str(tree))
    spec = importlib.util.spec_from_file_location(
        "hermes_lcm", tree / "__init__.py", submodule_search_locations=[str(tree)]
    )
    assert spec is not None
    package = importlib.util.module_from_spec(spec)
    sys.modules["hermes_lcm"] = package
    return (
        importlib.import_module("hermes_lcm.config"),
        importlib.import_module("hermes_lcm.engine"),
    )


def drain(engine_module: Any, timeout: float = 30.0) -> None:
    for name in ("_ROLLUP_MAINTENANCE_SCHEDULER", "_EMBEDDING_MAINTENANCE_SCHEDULER"):
        scheduler = getattr(engine_module, name, None)
        if scheduler is None:
            continue
        deadline = time.monotonic() + timeout
        with scheduler._condition:
            while (
                scheduler._queued_keys
                or scheduler._active_keys
                or scheduler._follow_up_key
            ) and time.monotonic() < deadline:
                scheduler._condition.wait(max(0.0, deadline - time.monotonic()))
            if (
                scheduler._queued_keys
                or scheduler._active_keys
                or scheduler._follow_up_key
            ):
                raise RuntimeError("scheduler not idle")


def run(
    seq: dict[str, Any],
    directory: Path,
    config_module: Any,
    engine_module: Any,
    catch: _Catch,
) -> dict[str, Any]:
    directory.mkdir(parents=True)
    catch.bad.clear()
    config = config_module.LCMConfig()
    config.database_path = str(directory / "lcm.db")
    engine = engine_module.LCMEngine(config=config, hermes_home=str(directory / "home"))
    counts: list[int] = []
    error = None
    try:
        engine.on_session_start(
            "S", platform="cli", context_length=128000, conversation_id="conv"
        )
        for view in seq["views"]:
            # Lifecycle failures are per-sequence errors too, not a driver abort.
            try:
                if view == "RESET":
                    engine.on_session_reset()
                elif view == "RESTART":
                    engine.shutdown()
                    drain(engine_module)
                    engine = engine_module.LCMEngine(
                        config=config, hermes_home=str(directory / "home")
                    )
                    engine.on_session_start(
                        "S", platform="cli", context_length=128000, conversation_id="conv"
                    )
                else:
                    engine.ingest(deepcopy([seq["rows"][row_id] for row_id in view]))
            except Exception as exc:
                error = type(exc).__name__
                break
            counts.append(len(engine._store.get_session_messages("S")))
        ledger: dict[str, int] = {}
        for row in engine._store.get_session_messages("S"):
            key = f"{row['role']}|{row['content']}"
            ledger[key] = ledger.get(key, 0) + 1
    finally:
        engine.shutdown()
        drain(engine_module)
    shutil.rmtree(directory)
    if catch.bad and error is None:
        error = "log:" + catch.bad[0]
    return {
        "seed": seq["seed"],
        "tag": seq.get("tag"),
        "counts": counts,
        "ledger": ledger,
        "err": error,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tree", type=Path)
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("workdir", type=Path)
    args = parser.parse_args()
    config_module, engine_module = load_tree(args.tree.resolve())
    catch = _Catch()
    logging.getLogger().addHandler(catch)
    try:
        if args.input == "-":
            for number, line in enumerate(sys.stdin, 1):
                result = run(
                    json.loads(line),
                    args.workdir / f"srv-{number}",
                    config_module,
                    engine_module,
                    catch,
                )
                print(json.dumps(result), flush=True)
        else:
            with open(args.input) as source, open(args.output, "w") as output:
                for line in source:
                    seq = json.loads(line)
                    result = run(
                        seq,
                        args.workdir / f"run-{seq['seed']}-{seq.get('tag') or 0}",
                        config_module,
                        engine_module,
                        catch,
                    )
                    output.write(json.dumps(result) + "\n")
                    output.flush()
    finally:
        logging.getLogger().removeHandler(catch)


if __name__ == "__main__":
    main()
