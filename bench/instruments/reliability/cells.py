"""The R1 cell registry: JSON-serialisable cells, glob selection and axis validation.

Tight tuning is the #553 probes' (tests/test_real_turn_loop_acp_override.py) scaled by window/64000, so a
128k cell has the same compaction cadence as the proven 64k probes. ``long-*`` cells use LCM's default
tuning with provider-reported usage, as that file's LONG cells do.
"""
from __future__ import annotations

import fnmatch
import os

BARS = ("B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9")
DRAIN_BARS = ("D1", "D2", "D3")  # scorers/drain.py: the host list drains under a hidden backlog (#597, #626)
TRANSPORTS = ("acp", "gateway")
FAULTS = {"crash_after_compaction_before_reply", "clean_exit_before_turn", "crash_mid_tool_call",
          "crash_after_rotation_before_child_row", "crash_between_session_end_and_start", "cancel_then_retry",
          "publication_failure", "plugin_switch", "forced_recovery", "p8_inject", "forced_compaction_then_turn"}
# issue -> (the bars that decide it, what an uncovered issue would need)
ISSUES = {
    7: (("B1", "B2"), ""),  # #493 (positional cursor misses an in-process rewrite of the last row) folded into #7
    # B5/B8 are downstream of #553 duplication in acp-history rotation (ref-0.21.5 vs customer-0.21.2,
    # nightly aa84e61d); re-check when #553 is fixed.
    553: (("B1", "B2", "B3", "B4", "B5", "B8"), ""), 561: (("B1", "B2"), ""), 563: (("B4",), ""),
    463: (("B7",), "Desktop/tui_gateway transport, >12k externalised user rows"),
    420: (("B4", "B5"), ""), 489: (("B1", "B2", "B4"), ""), 493: (("B1", "B2"), ""),
    496: (("B1", "B2"), "real gateway process with message timestamps rendered (gateway.message_timestamps.enabled)"),
    497: (("B6",), ""), 499: (("B1", "B2"), "fresh_tail 0 + objective-head merge + restart x3"),
    500: (("B7",), "native ON + dropped call + tool_call_id reuse"), 501: (("B1", "B2"), "fresh_tail 0 + restart x3"),
    503: (("B2",), ""), 534: (("B1", "B2"), ""),
    538: (("B1", "B2"), "merge behind the todo annotation"), 540: (("B1", "B2"), "folded carrier across rotation restart"),
    541: (("B1", "B2", "B4"), ""), 544: (("B1", "B2", "B4"), ""), 545: (("B1", "B2", "B3", "B4"), ""),
    546: (("B1", "B2", "B3", "B4"), ""), 547: (("B1", "B2", "B3", "B4"), ""), 549: (("B1", "B2"), ""),
    485: (("B2",), "upgrade from a pre-fix DB (R2)"), 542: (("B4",), "upgrade from a pre-#535 wedged DB (R2)"),
    559: (("B6", "B4"), ""), 566: (("B1", "B2", "B5"), ""),  # B5: a cross-lineage summary is recorded only there
    581: (("B3", "B4"), ""), 582: (("B8",), ""),  # native-on-off: every candidate event after the plugin switch
    597: (("D1", "D2"), ""), 626: (("D3",), ""),  # drain/hidden-backlog: data cells (ci.NON_GATE)
    # r34.4 and upstream-uid (2667c960), rotation: the host keeps a held composite AND its parts active in one
    # session; LCM stores the parts only
    821: (("B1", "B2", "B3", "B4", "B5", "B8"), "r34.4/upstream-uid host: held composite vs durable parts after a crash on rotation"),
    871: (("B2",), "r34.4 rotation: crash between parent end and child start, the crash turn's user row unstored"),
    # acp-process only in practice: the crash point races reply persistence (3 of 12 runs fail on either plugin ref);
    # the same two shapes occur on ref, customer, r34.4 and upstream-uid (not seen on upstream-main)
    861: (("B1", "B2"), "acp-process: intermittent around the crash after rotation"),
    916: (("F4",), "acp-process continuity: survival projection removes the current user tag"),
}
# issue -> host-name prefixes on which its bars are declared (absent: every host); ci.gate reads it
ISSUE_HOSTS = {821: ("r34.4-", "upstream-uid"), 861: ("ref-", "customer-", "r34.4-", "upstream-uid"), 871: ("r34.4-",)}
# issue -> transports on which its bars are declared (absent: every transport); ci.gate reads it
ISSUE_TRANSPORTS = {861: ("acp-process",)}


def tight(window: int) -> dict:
    k = window / 64000
    return {"LCM_CONTEXT_THRESHOLD": "0.5", "LCM_FRESH_TAIL_COUNT": "24", "LCM_FRESH_TAIL_MAX_TOKENS": str(int(12000 * k)),
            "LCM_LEAF_CHUNK_TOKENS": str(int(4000 * k)), "LCM_THRESHOLD_FULL_SWEEP_ENABLED": "true"}


def cell(cid, targets, *, in_place, transport="acp", turns=60, window=128000, repeat=None, native=False, user=None,
         assistant=None, tool_plan=(), faults=(), lcm_env=None, min_compactions=5, bars=None, doc="", **extra) -> dict:
    return {"id": cid, "targets": list(targets), "in_place": in_place, "native_recovery": native, "transport": transport,
            "turns": turns, "window": window,
            "user_text": {"repeat": repeat or int(400 * window / 64000), **(user or {})},
            "assistant": {"mode": "unique", "real_usage": False, "usage_scale": 1.0, **(assistant or {})},
            "tool_plan": list(tool_plan), "faults": list(faults),
            "lcm_env": {**(tight(window) if lcm_env is None else lcm_env),
                        **({"LCM_RELIABILITY_P8": os.environ["LCM_RELIABILITY_P8"]} if "LCM_RELIABILITY_P8" in os.environ else {})},
            "min_compactions": min_compactions,
            "final_compaction_check": True, "bars": list(bars or BARS), "doc": doc, **extra}


def modes():
    return (("in-place", True), ("rotation", False))


def continuity_cells() -> list[dict]:
    """R2-only diagnostics: ordinary turns and the existing multi-round tool plan."""
    out = []
    for m, ip in modes():
        out.append(cell(f"continuity/sole-user-tool-loop/{m}", [], in_place=ip, turns=1, min_compactions=2,
                        lcm_env={**tight(128000), "LCM_CONTEXT_THRESHOLD": "0.12", "LCM_FRESH_TAIL_COUNT": "4",
                                 "LCM_FRESH_TAIL_MAX_TOKENS": "2000", "LCM_LEAF_CHUNK_TOKENS": "1000"},
                        tool_plan=[{"turns": [1], "calls": [{"name": "read_file", "args": {
                            "path": f"{{files}}/big-{k:02d}.txt"}, "expect": {"min_chars": 10000}}]} for k in range(1, 13)],
                        big_files=12, continuity={"sole_user": True}, final_compaction_check=False,
                        doc="Diagnostic #900: one prompt, 12 distinct files/rounds because the host deduplicates identical reads; "
                            "15,360-token trigger, require two commits."))
        out.append(cell(f"continuity/survival-fit/{m}", [916], in_place=ip, turns=12, repeat=1600,
                        user={"repeat_from": {"9": 100}}, min_compactions=0,
                        lcm_env={**tight(128000), "LCM_CONTEXT_THRESHOLD": "0.12", "LCM_SURVIVAL_FIT": "true",
                                 "LCM_SURVIVAL_RESERVE": "0.9", "LCM_FRESH_TAIL_COUNT": "24"},
                        bars=[b for b in BARS if b != "B8"],
                        continuity={"require_survival_fit": True}, final_compaction_check=False,
                        doc="Diagnostic #916: low survival ceiling can project the newest user; F4 decides."))
        out.append(cell(f"continuity/survival-fit-older-turns/{m}", [], in_place=ip, turns=16, repeat=4000,
                        user={"repeat_from": {"9": 1}}, min_compactions=0,
                        lcm_env={**tight(128000), "LCM_CONTEXT_THRESHOLD": "0.8", "LCM_SURVIVAL_FIT": "true",
                                 "LCM_SURVIVAL_RESERVE": "0.5", "LCM_FRESH_TAIL_COUNT": "24",
                                 "LCM_FRESH_TAIL_MAX_TOKENS": "80000"},
                        bars=[b for b in BARS if b != "B8"],
                        continuity={"require_survival_fit": True}, final_compaction_check=False,
                        doc="Diagnostic #798: fit older whole turns with a moderate ceiling; newest short turns fit."))
        out.append(cell(f"continuity/forced-compaction-then-turn/{m}", [], in_place=ip, turns=20, min_compactions=0,
                        faults=[{"kind": "forced_compaction_then_turn", "turn": 19}],
                        continuity={"forced_followup": True}, final_compaction_check=False,
                        doc="Diagnostic: reuse ACP /compress after turn 19, then ordinary turn 20."))
    return out


# #659 probe nonces: letters G-Z only, so no hex-shaped token a host anchor index could harvest from the text.
PROBE_ENV = {**tight(128000), "LCM_CONTEXT_THRESHOLD": "0.12", "LCM_FRESH_TAIL_COUNT": "4",
             "LCM_FRESH_TAIL_MAX_TOKENS": "2000", "LCM_LEAF_CHUNK_TOKENS": "1000"}


def probe_cells() -> list[dict]:
    """#659 PR A, recorded only (R2, opt-in: never in ``--cells all``, so the nightly gate is unchanged). Request-time
    nonce probes, presence per compaction event: P1 host instruction (SOUL.md, rewritten before turn 16 with no
    restart), P2 open todos through the default ``tool_call`` bridge (a completed item is the control), P5 a constraint
    stated before the fresh tail (head, mid-text, another nonce shape) with a long tool turn after it."""
    v1, v2 = "QXWZKVJMTR", "ZWVKQJMXRT"
    soul = {"nonce": v1, "text": f"Reliability host note {v1}: a fixed line for a continuity check.",
            "rewrite": {"before_turn": 16, "nonce": v2, "text": f"Reliability host note {v2}: a fixed line for a continuity check."}}
    host = [{"probe": "host-instruction", "nonce": v1, "expect": "present", "carrier": "system", "once": True, "window": "before"},
            {"probe": "host-instruction", "nonce": v2, "expect": "absent", "window": "before"},
            {"probe": "host-instruction", "nonce": v2, "expect": "present", "carrier": "system", "once": True, "window": "after"},
            {"probe": "host-instruction", "nonce": v1, "expect": "absent", "window": "after"}]
    todo = {"in_progress": "PNWRQZHKTL", "pending": "SLYGXMTUVW", "completed": "KRWHZPYNQS"}
    todo_call = {"name": "tool_call", "args": {"name": "todo_list", "arguments": {"merge": False, "todos": [
        {"id": str(k), "content": f"probe item {n}", "status": st} for k, (st, n) in enumerate(todo.items(), 1)]}}}
    todo_probes = [{"probe": "todo-open" if st != "completed" else "todo-done-control", "nonce": n, "from_turn": 4,
                    "expect": "present" if st != "completed" else "absent", "carrier": "todo_fold"} for st, n in todo.items()]
    ctrl = "HWXQZTLGPN"
    reads = [{"turns": [6], "calls": [{"name": "read_file", "args": {"path": f"{{files}}/big-{k:02d}.txt"}}]} for k in (1, 2, 3)]
    out = []
    for m, ip in modes():
        common = dict(in_place=ip, turns=30, repeat=400, lcm_env=PROBE_ENV, min_compactions=2, final_compaction_check=False)
        out.append(cell(f"continuity/probe-host-instruction/{m}", [], **common, continuity={"soul": soul, "probes": host},
                        doc="#659 P1 (recorded; bar 100%): a SOUL.md nonce line in the system prompt after every commit; "
                            "the mid-run rewrite must appear only after the next commit (host-instruction-rebuild)."))
        out.append(cell(f"continuity/probe-todo/{m}", [], **common, tool_plan=[{"turns": [3], "calls": [todo_call]}],
                        continuity={"probes": todo_probes},
                        doc="#659 P2 (recorded): open todo items in the host's fold row after every commit, the completed "
                            "item never; its own cell because the fold merge shape is #538."))
        for shape, at, n in (("head", "head", "XQGTWZRMKY"), ("mid", "mid", "RVZLNWQHTX"), ("shape", "head", "gozu-ktvr-wyxn")):
            out.append(cell(f"continuity/probe-constraint-{shape}/{m}", [], **common, tool_plan=reads, big_files=3,
                            big_lines=60, user={"payload": {
                                "3": {"at": at, "text": f"CONSTRAINT {n}: answer in English and never touch the held file."},
                                "27": {"at": "head", "text": f"CONTROL {ctrl}: a late constraint inside the fresh tail."}}},
                            continuity={"probes": [{"probe": "constraint", "nonce": n, "expect": "present", "from_turn": 3},
                                                   {"probe": "constraint-control", "nonce": ctrl, "expect": "present",
                                                    "from_turn": 27}]},
                            doc=f"#659 P5 (recorded): a constraint nonce ({shape}) in turn 3, three large reads in turn 6; "
                                "the fake summariser keeps tags only, so this measures structural retention."))
    return out


def registry() -> list[dict]:
    crash = {"kind": "crash_after_compaction_before_reply"}
    group559 = [{"name": "read_file", "args": {"path": "{files}/small.txt"}},
                {"name": "lcm_expand", "args": {"store_id": 1, "max_tokens": 20000}, "expect": {"min_chars": 10000}}]
    cells = []
    for m, ip in modes():
        cells += [
            cell(f"overflow-recovery-restart/{m}", [534], in_place=ip, turns=36,
                 user={"repeat_from": {"31": 1, "32": 1600, "33": 800}}, bars=["B1", "B2"], min_compactions=1,
                 summary_repeat=200,
                 tool_plan=[{"turns": [32], "calls": [{"name": "write_file", "args": {
                     "path": "{files}/overflow.txt", "content": "alpha beta gamma delta " * 8000}}]}],
                 faults=[{"kind": "forced_recovery", "turn": 32, "cap": 8000},
                         {"kind": "crash_after_compaction_before_reply", "after_status": "overflow_recovery", "offset": 1},
                         {"kind": "clean_exit_before_turn", "after_restart": 1}],
                 doc="#534: prior v4 compaction, marked mid-tool overflow recovery, reply, then two cold restarts; "
                     "B1/B2 decide, B3/B4/B8 are diagnostics. No marked recovery means UNSUPPORTED."),
            cell(f"baseline/{m}/acp", [], in_place=ip, doc="Negative control: no faults, no tools; must PASS on main."),
            cell(f"acp-trailing/{m}", [483, 494], in_place=ip, user={"trailing_ws": True},
                 doc="ACP raw prompt with a trailing newline, persisted stripped (the #483/#494 persist rewrite)."),
            cell(f"repeat-identical-replies/{m}", [503], in_place=ip,
                 assistant={"mode": "repeat-identical", "repeat_turns": list(range(3, 61, 3))},
                 doc="Byte-identical replies on every third turn: only the multiset bar (B2) can see duplicates."),
            cell(f"crash-then-lcm-tool-in-merge-turn/{m}", [553, 563] + ([] if ip else [821]), in_place=ip, faults=[crash],
                 tool_plan=[{"turns": "restart", "calls": [{"name": "lcm_status", "args": {}}]}],
                 doc="#553 S1: crash after a preflight compaction; the merge turn dispatches a real LCM tool call."),
            cell(f"gateway-second-restart/{m}", [546, 547], in_place=ip, transport="gateway",
                 faults=[crash, {"kind": "clean_exit_before_turn", "after_restart": 3}],
                 doc="Crash after compaction, gateway restart, then a clean second restart (a deploy) 3 turns later."),
            cell(f"parallel-tool-group/{m}", [559, 497], in_place=ip,
                 tool_plan=[{"turns": list(range(2, 61, 3)), "calls": group559}],
                 doc="#559 shape: one assistant row calls read_file + lcm_expand (large result); lcm_expand ingests mid-turn."),
            cell(f"lcm-tool-mid-turn/{m}", [], in_place=ip,
                 tool_plan=[{"turns": list(range(4, 61, 4)), "calls": [{"name": "lcm_grep", "args": {"query": "alpha"}}]}],
                 doc="A single LCM tool call every fourth turn, no crash."),
            cell(f"cancel-retry/{m}", [7, 493, 544], in_place=ip, faults=[{"kind": "cancel_then_retry", "turn": 22}],
                 doc="ACP cancel (request_hard_interrupt) during the provider call, then the same prompt re-sent: the "
                     "host re-attaches the cancelled prompt (acp_adapter/server.py _attach_interrupted_prompt)."),
            cell(f"long-80/{m}", [], in_place=ip, turns=80, repeat=1000, lcm_env={},
                 assistant={"real_usage": True}, min_compactions=8,
                 doc="LCM default tuning with provider-reported usage, 80 turns (the LONG cells of the #553 probe file)."),
            cell(f"multi-session-one-process/{m}", [566], in_place=ip, cron_every=5,
                 doc="A chat session S0 and a fresh platform='cron' agent every 5 S0 turns in ONE host process "
                     "(cron/scheduler.py); cron turns are tagged K."),
        ]
        for ref in ("v0.23.3", "v0.24.3"):  # #581/#582: a native-ON store handed to the candidate with native OFF
            cells.append(cell(f"native-on-off/{ref}/{m}", [581, 582], in_place=ip, native=True, from_ref=ref,
                              faults=[{"kind": "plugin_switch", "turn": 31}], bars=["B3", "B4", "B8"],
                              doc=f"Turns 1-30 run lcm-x {ref} with native recovery ON; the candidate then takes over the "
                                  "same HERMES_HOME and store with native OFF. B3/B8 count every candidate-phase "
                                  "event after the plugin switch (pre_publication_counts: a diagnostic); B4 asks it "
                                  "to publish. B1/B2/B5-B7 are reported, not scored: the older "
                                  "ref's own phase decides them."))
        cells.append(cell(f"drain/hidden-backlog/{m}", [597, 626], in_place=ip, turns=70, min_compactions=2,
                          faults=[{"kind": "clean_exit_before_turn", "turn": 41}] if ip else [],
                          bars=list(DRAIN_BARS), drain={"phase2_turn": 41, "hold_seconds": 10.0},
                          doc="Data, not a gate (ci.NON_GATE): turns 1-40 store a backlog over >= 2 compactions, then "
                              "the host session ends (in-place: a clean host exit before turn 41 and an ACP restore; "
                              "rotation: the compaction rotation), so stored raw rows not yet summarized are no longer "
                              "in the host's list; turns 41-70 run on the new list. Per compaction the observer "
                              "records the list handed to compress, the list returned and how many host rows a leaf "
                              "replaced (scorers/drain.py D1-D3). Expected to FAIL on main: that is the measurement."))
        cells.append(cell(f"drain/hidden-backlog-large/{m}", [597, 626], in_place=ip, turns=180, repeat=100,
                          user={"repeat_from": {"151": 800}}, min_compactions=0,
                          lcm_env={**tight(128000), "LCM_CONTEXT_THRESHOLD": "0.99"},
                          faults=[{"kind": "clean_exit_before_turn", "turn": 151}], bars=list(DRAIN_BARS),
                          drain={"phase2_turn": 151, "hold_seconds": 10.0, "forget_host_rows": True,
                                 "phase2_lcm_env": {"LCM_CONTEXT_THRESHOLD": tight(128000)["LCM_CONTEXT_THRESHOLD"]}},
                          doc="Data, not a gate (ci.NON_GATE), acp-process only. Fixture B: turns 1-150 at LCM threshold "
                              "0.99 with short prompts (no compaction; ~300 raw stored rows), a clean host exit, then "
                              "the harness soft-archives the ACP session's active rows in the cell's state.db (the "
                              "host forgets its list; no host or plugin code changes), so every stored row is hidden. "
                              "Turns 151-180 run at the tight threshold with full-size prompts on the empty restored "
                              "list. D4 counts the phase-2 compactions with out == in (the plateau)."))
        for tr in ("acp-history", "gateway-reload"):
            cells.append(cell(f"crash-after-compaction/{m}/{tr}", [553, 561] + ([] if ip else [821]), in_place=ip,
                              transport="acp" if tr == "acp-history" else "gateway", faults=[crash],
                              doc="#553: os._exit inside the provider call of the first turn whose preflight compaction "
                                  "committed; the restart restores the dangling row and the next prompt merges into it."))
    cells += [
        cell("preflight-continue/in-place", [503], in_place=True, turns=40, user={"trailing_ws": True, "continue_turns": [17, 18, 19]},
             lcm_env={**tight(128000), "LCM_FRESH_TAIL_COUNT": "8"}, min_compactions=2,
             doc="Repeated 'continue' prompts under sub-threshold preflight ingest, fresh_tail 8."),
        cell("publication-failure/rotation-child", [541], in_place=False, turns=80, repeat=1000, lcm_env={},
             assistant={"real_usage": True}, faults=[{"kind": "publication_failure", "where": "rotation_child"}],
             bars=["B1", "B2", "B4"], min_compactions=0,
             doc="#541's bar: the FIRST rotation-child publication raises LifecyclePublicationConflictError "
                 "(generalises PROBE_CHILD_CONFLICT); the next child compaction must commit, with 0 parent copies. "
                 "B3/B5 do not apply to an injected conflict."),
        cell("publication-failure/rotation-child-persistent", [], in_place=False, turns=80, repeat=1000, lcm_env={},
             assistant={"real_usage": True}, bars=["B1", "B2", "B4"], min_compactions=0,
             faults=[{"kind": "publication_failure", "where": "rotation_child", "persistent": True}],
             doc="Data, not G-REL-1 (ci.NON_GATE): EVERY rotation-child publication raises, so no child compaction "
                 "can ever publish; a degraded-mode product question outside stabilization (DESIGN-436 REVISION 2 D-B)."),
        cell("publication-failure/pass-3-in-place", [541], in_place=True,
             faults=[{"kind": "publication_failure", "where": "pass_3"}], bars=["B1", "B2", "B4"], min_compactions=0,
             doc="The third publication attempt raises; later passes must recover."),
        cell("separator-heavy-retained/in-place", [545], in_place=True, user={"separator_turns": "all"}, faults=[crash],
             doc="Every prompt carries >=64 blank-line separators, so the dangling retained row does too (#545)."),
        cell("pressure-disagreement/in-place", [420], in_place=True, assistant={"real_usage": True, "usage_scale": 1.3},
             doc="The provider reports 1.3x the tokens actually sent (#420)."),
        cell("window-1m/in-place", [], in_place=True, window=1000000, turns=60,
             doc="1M window through engine.update_model; tight tuning and text volume scaled x15.6 from the 64k probes, "
                 "so LCM's threshold math runs at 1M (threshold 500k tokens, ~36k tokens per turn)."),
        cell("crash-after-rotation/rotation", [519, 549, 821, 861], in_place=False, user={"trailing_ws": True},
             faults=[{"kind": "crash_after_rotation_before_child_row"}],
             doc="os._exit right after the engine's rotation on_session_start, before any child row (#519/#549)."),
        cell("crash-between-end-and-start/rotation", [489, 821, 871], in_place=False,
             faults=[{"kind": "crash_between_session_end_and_start"}],
             doc="os._exit between on_session_end and on_session_start of a rotation (#489). UNSUPPORTED where the "
                 "host's rotation path never calls on_session_end."),
    ]
    cells += [cell(f"p8-control/{v}", [], in_place=True, bars=["B9"],
                   faults=[] if v == "none" else [{"kind": "p8_inject", "variant": v}],
                   doc="P8 flush positive control; none is the negative control.")
              for v in ("archived", "other-active", "random-snapshot", "none")]
    return cells


def validate(c: dict) -> None:
    assert c["transport"] in TRANSPORTS, c["id"]
    assert set(c["bars"]) <= set(BARS + DRAIN_BARS), c["id"]
    assert {f["kind"] for f in c["faults"]} <= FAULTS, c["id"]
    assert c["window"] in (128000, 1000000), c["id"]
    assert all(t in ISSUES or t in (483, 494, 519) for t in c["targets"]), c["id"]
    # Native recovery was removed (O2): only a native-on-off cell (an older lcm-x ref with it ON) may set it.
    assert not c["native_recovery"] or c.get("from_ref"), c["id"]
    for g in c["tool_plan"]:
        assert g["turns"] == "restart" or all(1 <= t <= c["turns"] for t in g["turns"]), c["id"]


def select(patterns: str, extra=(), opt_in=()) -> list[dict]:
    """``opt_in`` cells (R2 process_cell.R2_PROBE_CELLS) match an explicit pattern only, never ``all``."""
    cells = registry() + list(extra)  # extra: transport-only cells (R2 process_cell.R2_CELLS)
    for c in cells + list(opt_in):
        validate(c)
    if patterns.strip() == "all":
        return cells
    pats = [p.strip() for p in patterns.split(",") if p.strip()]
    chosen = [c for c in cells + list(opt_in) if any(fnmatch.fnmatchcase(c["id"], p) for p in pats)]
    if not chosen:
        raise ValueError(f"no cell matches {patterns!r}")
    return chosen
