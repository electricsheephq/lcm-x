"""Catch-up grid runner: walk synthetic cells on ONE engine tree and write one JSON record per pass.

    python3 -I -B catchup_grid_runner.py --tree <tree> --cells <json> --out <json> [--syspath <dir> ...]
                                         [--patch <file>] [--workdir <dir>]

It points HOME and HERMES_HOME under ``--workdir`` and drops every LCM_/HERMES_ variable before the engine
import. Run it isolated (``-I``): the engine is imported ONLY from ``--tree`` (registered as ``hermes_lcm`` the way
tests/conftest.py does), and ``--syspath`` adds the host stub and dependency directories, so two trees never share
a process. The clock is simulated only: ``time.monotonic`` is a counter that the stand-in summariser and the
injected step costs advance. The summariser honours the timeout it is handed: a call longer than its timeout ends
at the timeout as a timeout error. Pre-call work is added once per pass after the compress-side retained-anchor
step, ``delta`` after every stored leaf, ``finalize`` before the stop line of a pass that made a call.

Growth between passes is ``growth_turns`` turns plus whole turns until ``growth_tokens``. Steady cells hand out a
summary of ``summary_tokens`` tokens; ``summary_area`` is the tokens of the verified summary heads in the returned list.

A cell with ``no_system_row`` gives the host view no leading system row. Pointers (survival-fit notices, projected
stubs) count only when they are in the RETURNED list: loss_check parses them from ``result``, never from the store.

The per-pass loss oracle reads the store rows and the DAG, not the engine's frontier bookkeeping: every stored row
that is not under a node whose verified summary head is in the returned list must be in the returned list by
content (its own row, or the text carried behind a summary head), tool calls and results stay paired, and a pass
with no call, no new node and an unchanged list returns the identical object. ``--patch`` executes a file in a
namespace holding the imported modules, after import and before the first cell (the test's negative variant).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import re
import importlib.util
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

PAD_UNIT = " alpha beta gamma delta"
SUMMARY = "Earlier turns.\nExpand for details about: turns"
CLOCK_START = 10_000.0
# Notes assembly appends to the stored system prompt: the LCM note, and the survival fit's dropped-rows notice.
SYSTEM_NOTES = ("\n\n[Note: This conversation uses Lossless Context Management (LCM).", "\n\n[LCM survival fit: ")
SURVIVAL_PROJECTED = "[LCM survival fit: this "  # a row replaced by its store id (survival_codec)
# Pointers a survival fit leaves for rows it moved out of the list: its notice names a store id range (survival_fit.py
# _NOTICE), and a projected row names its own store id (survival_codec.py _PROJECTED).
NOTICE_RANGE = re.compile(r"\[LCM survival fit: \d+ earlier messages \(store ids (\d+)\.\.(\d+)\)")
# The cumulative notice of the round-5 N1 arm: one exact call per range (it counts only when the call is the one
# that loads that range: this session, after_store_id = first - 1).
N1_CALL = re.compile(r'store ids (\d+)\.\.(\d+): lcm_load_session\(session_id="([^"]+)", after_store_id=(\d+), limit=200\)')
PROJECTED_ID = re.compile(r"\[LCM survival fit: this [a-z_]+ message \(\d+ tokens\) is stored verbatim as store id (\d+)")
_MODULES = ("config", "dag", "engine", "compaction", "escalation", "survival_fit", "tokens")


class SimClock:
    """Simulated monotonic clock: no real time in it."""

    def __init__(self):
        self.offset = CLOCK_START

    def __call__(self) -> float:
        return self.offset


def load_tree(tree: Path, extra_paths: list[str]) -> dict:
    sys.dont_write_bytecode = True  # never write __pycache__ into a tree
    for entry in extra_paths:
        if entry not in sys.path:
            sys.path.append(entry)
    spec = importlib.util.spec_from_file_location(
        "hermes_lcm", str(tree / "__init__.py"), submodule_search_locations=[str(tree)])
    package = importlib.util.module_from_spec(spec)
    package.__path__ = [str(tree)]
    package.__package__ = "hermes_lcm"
    sys.modules["hermes_lcm"] = package  # not executed: it registers with the host
    return {name: importlib.import_module(f"hermes_lcm.{name}") for name in _MODULES}


class Provider:
    """Stand-in summariser: ``script`` seconds per call index within a pass; ``None`` hangs to its timeout."""

    def __init__(self, clock: SimClock, script, summary: str = SUMMARY):
        self.clock, self.script, self.summary = clock, script, summary
        self.reset_pass()

    def reset_pass(self):
        self.started = self.cut = 0
        self.cut_flags: list = []
        self.timeouts: list = []
        self.kinds: list = []

    def seconds(self, index: int):
        script = self.script if isinstance(self.script, list) else [self.script]
        value = script[min(index, len(script) - 1)]
        return None if value == "hang" else float(value)

    def __call__(self, prompt, max_tokens, model="", timeout=None, reasoning_effort=""):
        took = self.seconds(self.started)
        self.started += 1
        self.timeouts.append(None if timeout is None else round(float(timeout), 3))
        self.kinds.append("leaf" if "user turn" in prompt else "condensation")
        self.cut_flags.append(bool(took is None or (timeout is not None and took > timeout)))
        if took is None or (timeout is not None and took > timeout):
            self.cut += 1
            self.clock.offset += float(timeout) if timeout is not None else 600.0
            raise TimeoutError("synthetic provider timed out")
        self.clock.offset += took
        return self.summary


def sized_summary(tokens: int) -> str:
    """A summary of about ``tokens`` tokens (the steady cells' leaf and condensation summaries)."""
    return "Earlier turns.\n" + "synthetic fact " * max(1, (tokens * 4) // 15) + "\nExpand for details about: turns"


def _pad(tokens: int) -> str:
    return PAD_UNIT * max(1, (tokens * 4) // len(PAD_UNIT))


def make_turns(start: int, count: int, row_tokens: int, tool_every: int) -> list[dict]:
    rows = []
    for index in range(start, start + count):
        tag = f"T{index:05d}"
        rows.append({"role": "user", "content": f"[{tag}] user turn{_pad(row_tokens)}", "timestamp": 10.0 * (index + 1)})
        if tool_every and index % tool_every == 0:
            rows.append({"role": "assistant", "content": f"running tool for {tag}", "tool_calls": [
                {"id": f"call-{tag}", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}]})
            rows.append({"role": "tool", "tool_call_id": f"call-{tag}", "content": f"[{tag}] result{_pad(row_tokens)}"})
        rows.append({"role": "assistant", "content": f"reply to {tag}{_pad(row_tokens)}"})
    return rows


def _row_key(role, content) -> str:
    text = content if isinstance(content, str) else json.dumps(content, sort_keys=True)
    return hashlib.sha256(f"{role}\x00{text}".encode()).hexdigest()


def _covered_rows(engine, node_ids: set) -> set:
    covered, stack, seen = set(), list(node_ids), set()
    while stack:
        node_id = stack.pop()
        if node_id in seen:
            continue
        seen.add(node_id)
        node = engine._dag.get_node(int(node_id))
        if node is None:
            continue
        if node.source_type == "messages":
            covered.update(int(store_id) for store_id in node.source_ids)
        else:
            stack.extend(int(child) for child in node.source_ids)
    return covered


def loss_check(engine, view_in: list, result: list, did_work: bool) -> dict:
    """The A0 oracle for one pass (see the module docstring)."""
    returned, heads, fit_notice, projected = Counter(), set(), False, 0
    ranges, pointed = [], set()
    for row in result:
        role, content = row.get("role"), row.get("content")
        if isinstance(content, str):
            ranges += [(int(a), int(b)) for a, b in NOTICE_RANGE.findall(content)]
            ranges += [(int(a), int(b)) for a, b, session, after in N1_CALL.findall(content)
                       if session == engine._session_id and int(after) == int(a) - 1]
            pointed.update(int(x) for x in PROJECTED_ID.findall(content))
        if role == "user" and isinstance(content, str):
            end, node_ids = engine._verified_lcm_summary_prefix(content)
            if end is not None:
                heads.update(node_ids)
                carried = engine._generated_context_carrier_remainder(row)
                if carried is not None:
                    returned[_row_key("user", carried)] += 1
                continue
        if role == "system" and isinstance(content, str):
            for note in SYSTEM_NOTES:
                if note in content:
                    content = content.split(note, 1)[0]  # the stored prompt, before the notes assembly appends
        projected += isinstance(content, str) and content.startswith(SURVIVAL_PROJECTED)
        returned[_row_key(role, content)] += 1
    fit_notice = bool(ranges)
    covered = _covered_rows(engine, heads)
    stored_rows = engine._store.get_session_messages(engine._session_id, limit=10_000_000)
    keys = [_row_key(stored.get("role"), stored.get("content")) for stored in stored_rows]
    # Content-aware covered side: a byte-equal copy of a row under a covered node counts as covered.
    covered_keys = {key for stored, key in zip(stored_rows, keys) if int(stored["store_id"]) in covered}
    figures = {}
    for basis in ("content", "id"):
        missing, needed, behind = [], Counter(), []
        for stored, key in zip(stored_rows, keys):
            if int(stored["store_id"]) in covered or (basis == "content" and key in covered_keys):
                continue
            needed[key] += 1
            if needed[key] > returned[key]:
                store_id = int(stored["store_id"])
                if store_id in pointed or any(first <= store_id <= last for first, last in ranges):
                    behind.append(store_id)  # A0b: out of the list, but a survival-fit pointer names it
                    continue
                missing.append({"store_id": store_id, "role": stored.get("role"),
                                "head": str(stored.get("content") or "")[:40]})
        figures[basis] = (missing, behind)
    (missing, behind), (missing_by_id, behind_by_id) = figures["content"], figures["id"]
    calls, results, unpaired = set(), [], []
    for index, row in enumerate(result):
        for call in row.get("tool_calls") or []:
            calls.add(call.get("id"))
        if row.get("role") == "tool":
            results.append(row.get("tool_call_id"))
            if row.get("tool_call_id") not in calls:
                unpaired.append({"index": index, "result_without_call": row.get("tool_call_id")})
    unpaired.extend({"call_without_result": call_id} for call_id in sorted(calls - set(results), key=str))
    unchanged = result == view_in
    return {"missing": missing[:20], "missing_count": len(missing), "unpaired": unpaired[:20],
            "noop_not_identical": bool(not did_work and unchanged and result is not view_in),
            "noop_changed": bool(not did_work and not unchanged), "heads": len(heads),
            "survival_fit_notice": fit_notice, "survival_projected_rows": projected,
            "behind_notice_count": len(behind), "behind_notice_ids": [behind[0], behind[-1]] if behind else [],
            "missing_count_by_id": len(missing_by_id), "behind_notice_count_by_id": len(behind_by_id),
            # rows neither in the list nor under a summary shown in it, pointer or not
            "unshown_rows": len(missing) + len(behind)}


def uncondensed_per_level(engine) -> dict:
    """Summary nodes per depth that no higher node condenses (the assembly's candidates per level)."""
    nodes = engine._dag.get_session_nodes(engine._session_id, limit=10_000_000)
    children = {int(child) for node in nodes if node.source_type != "messages" for child in node.source_ids}
    counts = Counter(int(node.depth) for node in nodes if int(node.node_id) not in children)
    return {str(depth): counts[depth] for depth in sorted(counts)}


def duplicate_rows(engine) -> int:
    """Stored rows of the session whose role and content another stored row already has."""
    keys = [_row_key(row.get("role"), row.get("content"))
            for row in engine._store.get_session_messages(engine._session_id, limit=10_000_000)]
    return len(keys) - len(set(keys))


def _wrap(engine, name: str, before=None, after=None):
    original = getattr(engine, name)

    def wrapped(*args, **kwargs):
        if before is not None:
            before(*args, **kwargs)
        result = original(*args, **kwargs)
        if after is not None:
            after(*args, **kwargs)
        return result

    setattr(engine, name, wrapped)


def _seed_prefix(mods, engine, cell, target: int) -> None:
    shape = cell.get("prefix", "none")
    total = {"none": 0, "under": target // 2, "1-2x": target * 3 // 2,
             "over2x": target * 5 // 2, "over2x_nogroup": target * 5 // 2}[shape]
    if not total:
        return
    depths = [0, 1, 2, 3] if shape == "over2x_nogroup" else [0] * max(2, int(engine._config.condensation_fanin))
    for index, depth in enumerate(depths):
        engine._dag.add_node(mods["dag"].SummaryNode(
            session_id=engine._session_id, depth=depth, summary=f"arc {index} " + "history " * 40,
            token_count=total // len(depths), source_token_count=2 * total // len(depths), source_ids=[],
            source_type="messages", created_at=float(index + 1)))


def run_cell(mods, clock: SimClock, cell: dict, workdir: Path) -> list[dict]:
    window = int(cell["window"])
    settings = {"large_output_externalization_enabled": False,
                "large_output_active_replay_stubbing_enabled": False, "temporal_rollups_enabled": False,
                "fresh_tail_count": 8, "leaf_chunk_tokens": int(cell.get("chunk", 24_000)),
                "context_threshold": 0.5, "threshold_full_sweep_enabled": True, "max_assembly_tokens": window,
                "l3_truncate_tokens": 2, "summary_timeout_ms": 120_000,
                "database_path": str(workdir / f"{cell['id']}.db"), **cell.get("settings", {})}
    engine = mods["engine"].LCMEngine(config=mods["config"].LCMConfig(**settings))
    engine.on_session_start("S", platform="telegram", context_length=window, conversation_id="conv")
    if cell.get("depth_ratio"):
        engine._config.depth_target_ratio = float(cell["depth_ratio"])
    summary = sized_summary(int(cell["summary_tokens"])) if cell.get("summary_tokens") else SUMMARY
    provider = Provider(clock, cell["call"], summary)
    mods["escalation"]._invoke_summary_llm = provider
    route = engine._primary_summary_route()
    estimates = engine.__dict__.setdefault("_foreground_estimates", mods["escalation"].ForegroundEstimates())
    if cell.get("estimate") is not None:
        estimates.record_call(route, float(cell["estimate"]))
    in_pass = {"active": False, "pre_done": False}

    def pre_work(*_args, **_kwargs):
        if in_pass["active"] and not in_pass["pre_done"]:
            in_pass["pre_done"] = True
            clock.offset += float(cell.get("pre", 0))

    def finalize(budget, *_args, **_kwargs):
        in_pass["budget"] = budget
        if budget.last_call_ended is not None:
            clock.offset += float(cell.get("finalize", 0))

    _wrap(engine, "_prepare_retained_user_anchor", after=pre_work)
    _wrap(engine, "_publish_leaf_node", after=lambda *a, **k: setattr(clock, "offset", clock.offset + float(cell.get("delta", 0))))
    _wrap(engine, "_finish_foreground_budget", before=finalize)
    sweep_target = int(engine._config.summary_prefix_target_tokens) or min(
        int(engine._config.leaf_chunk_tokens), int(engine.threshold_tokens * 0.25))
    _seed_prefix(mods, engine, cell, sweep_target)
    count = mods["tokens"].count_messages_tokens
    row_tokens, tool_every = int(cell.get("row_tokens", 500)), int(cell.get("tool_every", 3))
    # no_system_row: the host view starts with the first conversation row (issue 2896: the reference host has none).
    view = [] if cell.get("no_system_row") else [{"role": "system", "content": "system prompt"}]
    view += make_turns(0, int(cell["raw_tokens"]) // (2 * row_tokens), row_tokens, tool_every)
    next_turn = len([row for row in view if row["role"] == "user"])
    fixed = max(0, engine.threshold_tokens + int(cell.get("over_tokens", 0)) - count(view))
    kind = cell.get("kind", "auto")
    records, leaves_cum, drained_cum, summary_drained_cum, result = [], 0, 0, 0, None
    count_text = mods["tokens"].count_tokens
    max_passes = int(cell.get("passes", 10))
    try:
        for index in range(max_passes):
            if index:
                grown = make_turns(next_turn, int(cell.get("growth_turns", 0)), row_tokens, tool_every)
                next_turn += int(cell.get("growth_turns", 0))
                while count(grown) < int(cell.get("growth_tokens", 0)):  # whole turns until the token growth
                    grown += make_turns(next_turn, 1, row_tokens, tool_every)
                    next_turn += 1
                view = list(result) + grown
                clock.offset += float(cell.get("turn_gap", 60))
            prompt = count(view) + fixed
            if kind == "overflow":
                prompt = max(prompt, window + 1)
            engine.ingest(view)
            triggered = kind == "forced" or engine.should_compress(prompt)
            nodes_before = {node.node_id: node.depth for node in engine._dag.get_session_nodes("S", limit=10_000_000)}
            status_before = engine._last_threshold_full_sweep
            hold_before = (float(getattr(engine, "_sweep_budget_hold_until", 0) or 0), engine._no_progress_hold)
            provider.reset_pass()
            record = {"tree": "", "cell": cell["id"], "pass": index, "triggered": bool(triggered),
                      "prompt_before": prompt,
                      # reached the gate: triggered, or at or over the trigger and refused (a hold)
                      "gate_reached": bool(triggered or prompt >= engine.threshold_tokens)}
            if not triggered:
                result = view
                record.update(calls_started=0, calls_cut=0, call_timeouts=[], call_cut_flags=[], call_kinds=[], leaves=0,
                              condensations=0, stop_reason="not_triggered", status="", wall=0.0, pre_seen=None,
                              prompt_after=prompt, returned_is_input=True, loss=None, summary_drained=0)
            else:
                in_pass.update(active=True, pre_done=False, budget=None)
                started = clock.offset
                try:
                    result = engine.compress(view, current_tokens=prompt, force=kind == "forced")
                finally:
                    in_pass["active"] = False
                wall = clock.offset - started
                nodes_after = {node.node_id: node.depth
                               for node in engine._dag.get_session_nodes("S", limit=10_000_000)}
                new = [depth for node_id, depth in nodes_after.items() if node_id not in nodes_before]
                # tokens removed by this pass's stored nodes only (source tokens less summary tokens), never by a fit cut
                fresh_nodes = [engine._dag.get_node(int(node_id)) for node_id in nodes_after if node_id not in nodes_before]
                record["summary_drained"] = sum(max(0, int(node.source_token_count) - int(node.token_count))
                                                for node in fresh_nodes if node is not None)
                status = engine._last_threshold_full_sweep
                fresh = status is not status_before and isinstance(status, dict)
                budget = in_pass.get("budget")
                pre_seen = (None if budget is None or budget.first_call_started is None
                            else round(budget.first_call_started - budget.t0, 3))
                record.update(
                    calls_started=provider.started, calls_cut=provider.cut, call_timeouts=provider.timeouts,
                    call_cut_flags=provider.cut_flags,
                    call_kinds=provider.kinds, leaves=sum(1 for depth in new if depth == 0),
                    condensations=sum(1 for depth in new if depth > 0),
                    stop_reason=str(status.get("stop_reason", "")) if fresh else "",
                    status=str(engine.last_compression_status or ""), wall=round(wall, 3), pre_seen=pre_seen,
                    prompt_after=count(result) + fixed, returned_is_input=result is view,
                    loss=loss_check(engine, view, result, did_work=bool(provider.started or new)))
            leaves_cum += record["leaves"]
            drained_cum += record["prompt_before"] - record["prompt_after"]
            summary_drained_cum += record["summary_drained"]
            breaker = engine._summary_circuit_breaker.route_status([route])
            hold_until = float(getattr(engine, "_sweep_budget_hold_until", 0) or 0)
            record["duplicate_rows"] = duplicate_rows(engine)
            record["uncondensed"] = uncondensed_per_level(engine)
            record["hold_armed"] = bool(hold_until > hold_before[0] or (
                engine._no_progress_hold is not None and engine._no_progress_hold != hold_before[1]))
            heads = [engine._verified_lcm_summary_prefix(row["content"])[0]
                     if row.get("role") == "user" and isinstance(row.get("content"), str) else None for row in result]
            record.update(
                trigger=int(engine.threshold_tokens), drained_cum=drained_cum, summary_drained_cum=summary_drained_cum, leaf_summary_tokens=count_text(summary), leaves_cum=leaves_cum,
                raw_rows_after=sum(1 for row, end in zip(result, heads) if row.get("role") != "system" and end is None),
                summary_area=sum(count_text(row["content"][:end]) for row, end in zip(result, heads) if end),
                frontier_tokens=int(engine._summary_frontier_tokens()),
                circuit={"state": breaker.get("state"), "last_error_class": breaker.get("last_error_class")},
                holds={"sweep_hold_left": round(max(0.0, hold_until - clock.offset), 3) if hold_until else 0.0,
                       "sweep_hold_turn_paced": bool(getattr(engine, "_sweep_budget_hold_turn_paced", False)),
                       "no_progress_hold": engine._no_progress_hold is not None,
                       "hung_strike": engine._hung_summary_strike is not None})
            records.append(record)
            engine.note_turn_complete()
            if cell.get("until_drained") and not engine.should_compress(count(result) + fixed):
                break
    finally:
        engine.shutdown()
    return records


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tree", required=True)
    parser.add_argument("--cells", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--workdir")
    parser.add_argument("--syspath", action="append", default=[])
    parser.add_argument("--patch")
    parser.add_argument("--describe", action="store_true", help="write only the tree header")
    args = parser.parse_args(argv)
    tree = Path(args.tree).resolve()
    workdir = Path(args.workdir or Path(args.out).parent / "work")
    home = workdir / "home"
    home.mkdir(parents=True, exist_ok=True)
    for key in [key for key in os.environ if key.startswith(("LCM_", "HERMES_"))]:
        del os.environ[key]  # no product setting or host home reaches the engine
    os.environ["HOME"], os.environ["HERMES_HOME"] = str(home), str(home / ".hermes")  # before any engine import
    clock = SimClock()
    time.monotonic = clock  # before any engine import: modules read time.monotonic through the module
    mods = load_tree(tree, args.syspath)
    mods["survival_fit"]._host_estimate = lambda messages: None  # the char counter, as the suite pins it (#615)
    tokens = mods["tokens"]
    tokens.count_tokens = lambda text: 0 if not text else (
        tokens._fallback_token_estimate(text) if isinstance(text, str) else len(text) // 4 + 1)
    header = {"tree": str(tree),
              "catchup_constant": any(hasattr(mods[name], "_CATCHUP_HARD_SECONDS")
                                      for name in ("compaction", "escalation")),
              "modules": sorted({str(Path(module.__file__).resolve()) for name, module in sys.modules.items()
                                 if name.startswith("hermes_lcm.") and getattr(module, "__file__", None)}),
              "patched": bool(args.patch), "home": os.environ["HOME"], "hermes_home": os.environ["HERMES_HOME"]}
    output = {"header": header, "records": [], "cell_seconds": {}}
    if not args.describe:
        if args.patch:
            namespace = {"mods": mods, "clock": clock, **mods}
            exec(compile(Path(args.patch).read_text(), args.patch, "exec"), namespace)
        for cell in json.loads(Path(args.cells).read_text()):
            began = time.perf_counter()
            records = run_cell(mods, clock, cell, workdir)
            for record in records:
                record["tree"] = str(tree)
            output["records"].extend(records)
            output["cell_seconds"][cell["id"]] = round(time.perf_counter() - began, 3)
        header["modules"] = sorted({str(Path(module.__file__).resolve()) for name, module in sys.modules.items()
                                    if name.startswith("hermes_lcm.") and getattr(module, "__file__", None)})
    Path(args.out).write_text(json.dumps(output, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
