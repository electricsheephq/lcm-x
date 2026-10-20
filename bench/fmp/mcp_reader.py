"""Stdio MCP server over ONE frozen FMP store: the agentic toolset, outputs unchanged.

Speaks newline-delimited JSON-RPC 2.0 (MCP stdio). It opens the store with the same
query-only delivery open as in-process agentic runs and never builds or migrates one.
Every call is logged as metadata only (tool, argument hash, latency, output size).
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "bench.fmp"

from .common import load_lcm, quiet, sha, utc  # noqa: E402

PROTOCOL = "2025-06-18"
OVER_BUDGET = json.dumps({"error": "tool budget exhausted; answer now"})


class Reader:
    def __init__(self, spec, engine, offered, log):
        self.spec, self.engine, self.offered, self.log = spec, engine, offered, Path(log)
        self.counted = 0

    def tools(self):
        return [{"name": s["function"]["name"], "description": s["function"].get("description", ""),
                 "inputSchema": s["function"].get("parameters") or {"type": "object", "properties": {}}}
                for s in self.offered]

    def call(self, name, arguments):
        from . import delivery
        started = time.monotonic()
        counted = self.counted < self.spec["max_calls"]
        if counted:
            with delivery.question_clock(self.spec["question"]["question_date"]):  # D12
                raw = delivery.dispatch(name, arguments if arguments is not None else {}, self.engine, self.offered)
            self.counted += 1
        else:
            raw = OVER_BUDGET
        try:
            error = "error" in json.loads(raw)
        except (ValueError, TypeError):
            error = False
        with self.log.open("a") as stream:
            stream.write(json.dumps({"at": utc(), "tool": name, "counted": counted, "error": error,
                "args_sha256": sha(json.dumps(arguments, sort_keys=True).encode()),
                "latency_s": time.monotonic() - started, "output_bytes": len(raw.encode())}) + "\n")
        return raw, error

    def handle(self, request):
        method, ident = request.get("method"), request.get("id")
        if ident is None:  # notifications need no reply
            return None
        if method == "initialize":
            result = {"protocolVersion": (request.get("params") or {}).get("protocolVersion") or PROTOCOL,
                      "capabilities": {"tools": {"listChanged": False}},
                      "serverInfo": {"name": "lcmx-fmp", "version": "1"}}
        elif method == "tools/list":
            result = {"tools": self.tools()}
        elif method == "tools/call":
            params = request.get("params") or {}
            raw, error = self.call(params.get("name"), params.get("arguments"))
            result = {"content": [{"type": "text", "text": raw}], "isError": error}
        elif method == "ping":
            result = {}
        else:
            return {"jsonrpc": "2.0", "id": ident, "error": {"code": -32601, "message": "method not found"}}
        return {"jsonrpc": "2.0", "id": ident, "result": result}


def serve(spec, stdin, stdout):
    from . import datasets, delivery
    from .stores import Stores
    if spec["config"].partition("+rerank:")[0] == "voyage" and not os.environ.get("VOYAGE_API_KEY"):
        # Read only in this trusted process, never in the model CLI or its MCP configuration.
        os.environ["VOYAGE_API_KEY"] = subprocess.run(
            ["security", "find-generic-password", "-s", "VOYAGE_API_KEY", "-w"],
            capture_output=True, text=True, check=True).stdout.strip()
    lme = load_lcm(spec["repo"])
    question = datasets.Question(spec["question"]["qid"], spec["question"]["dataset"], "", "", "",
                                 spec["question"]["question_date"], False, [], spec["question"]["store_id"],
                                 spec["question"]["session_map"])
    stores = Stores(spec["store_root"], spec["config"], spec["repo"], spec["dataset_sha"], lme,
                    spec["summaries"], spec.get("embed_cache"))
    with stores.engine(question) as (engine, _):
        reader = Reader(spec, engine, delivery.schemas(engine), spec["log"])
        for line in stdin:
            if not line.strip():
                continue
            try:
                reply = reader.handle(json.loads(line))
            except ValueError:
                reply = {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "parse error"}}
            if reply is not None:
                stdout.write(json.dumps(reply, ensure_ascii=False) + "\n")
                stdout.flush()


def main():
    spec = json.loads(Path(sys.argv[1]).read_text())
    quiet()
    # Only JSON-RPC reaches the client: dependency output goes to /dev/null, never to the harness.
    protocol = os.fdopen(os.dup(1), "w", encoding="utf-8")
    sink = os.open(os.devnull, os.O_WRONLY)
    os.dup2(sink, 1)
    os.dup2(sink, 2)
    sys.stdout = sys.stderr = open(os.devnull, "w")
    try:
        serve(spec, sys.stdin, protocol)
    except Exception as exc:
        with Path(spec["log"]).open("a") as stream:
            stream.write(json.dumps({"at": utc(), "fatal": type(exc).__name__}) + "\n")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
