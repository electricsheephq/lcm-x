"""#2856 whole-session differential: the candidate tree and a base tree produce identical per-step dumps.

Set LCMX_FIXEDWORK_BASE_TREE to a checkout of the base (the branch point); skipped when unset. Both trees load as
separate packages in this process (tests/differential/paired.py's loader). Each seeded synthetic session runs
ordinary turns, forced compaction passes (a stand-in summariser that honours its timeout), host-side impure rows
(pruned, out-of-band, persisted-output, externalized, ingest payload, survival projection, summary carriers,
near misses), a rebind, a session switch, a restart on the same store and session end. A step dump holds every
row of every table of the database, every payload file, the ingest cursor, the status and every list a pass
returns. The clock (time.time, time_ns, gmtime, localtime) is frozen per step, and each tree runs at the same path (the run directory is renamed away
after each run), so wall-clock and path values are equal by construction; the store's background FTS threads are joined before
every operation and dump (they otherwise race writes, in base-vs-base runs too). Every pass also checks, on both trees,
that each stored row of the session not under a DAG node whose summary is in the returned list appears in it.
Imports no memo symbol.
"""
import base64
import hashlib
import importlib
import json
import os
import random
import re
import sqlite3
import tempfile
import secrets
import time
import uuid
from copy import deepcopy
from collections import Counter
from pathlib import Path

import pytest

from tests.differential.paired import _load

BASE = os.environ.get("LCMX_FIXEDWORK_BASE_TREE")
CANDIDATE = Path(__file__).resolve().parents[2]
SEEDS = int(os.environ.get("LCMX_FIXEDWORK_SEEDS", "30"))
WORD = " lorem ipsum dolor"
OOB = ("\n\n[OUT-OF-BAND USER MESSAGE — a direct message from the user, delivered mid-turn; not tool output]\n"
       "check the 2026 logs instead\n[/OUT-OF-BAND USER MESSAGE]")
PRUNED = "[Old tool output cleared to save context space]"
NEAR = ["[LCM survival fit: that", "[Externalized", "<persisted-output", "[OUT-OF-BAND USER MESSAG",
        "[Expand for details", "lcm_survival_fi"]
CTIME = re.compile(r'(ctime_ns(?:=|": ?))\d+')
# Literal markers of each impure kind in a host row's content or calls (the tally only; the gate is reconcile's).
KIND_MARKERS = (("projection", "[LCM survival fit: this "), ("pruned", PRUNED), ("oob", "[OUT-OF-BAND USER MESSAGE"),
                ("persisted", "<persisted-output>"), ("externalized", "[Externalized tool output"),
                ("ingest_placeholder", "[Externalized LCM ingest payload:"), ("summary", "[Expand for details:"),
                ("near_miss", "lcm_survival_fi"))
SUMMARY_EXPAND = "[Expand for details:"
PROJECTION_NAMES = re.compile(
    r"\[LCM survival fit: this [a-z_]+ message \(\d+ tokens\) is stored verbatim as store id (\d+);")
SUMMARY_PART = re.compile(r"\[[A-Za-z0-9 -]+ Summary \(d(\d+), node (\d+)\)\]")

pytestmark = pytest.mark.skipif(not BASE, reason="LCMX_FIXEDWORK_BASE_TREE is not set")


def _summariser(prompt, max_tokens, model="", timeout=None, reasoning_effort=""):
    assert timeout is None or timeout > 0
    return f"Earlier turns ({len(prompt) % 97}).\nExpand for details about: turns"


def _persisted(home, cid, text):
    path = Path(home) / "cache" / "spillover" / f"{cid}.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.encode("utf-8"))
    os.utime(path, ns=(1_700_000_000 * 10 ** 9,) * 2)  # a fixed mtime; the inode ctime is normalised in the dump
    return ("<persisted-output>\n"
            f"This tool result was too large ({len(text):,} characters, {len(text) / 1024:.1f} KB).\n"
            f"Full output saved to: {path}\n"
            "Use the read_file tool with offset and limit to access specific sections of this output.\n"
            "Recovery: page through the saved file with read_file (offset/limit) or "
            "process it with execute_code — do NOT re-request the same data from the "
            "remote API; the full result is already on disk.\n\n"
            f"Preview (first 30 chars):\n{text[:30]}\n...\n</persisted-output>")


class Host:
    """A host transcript: appends rows (merging consecutive user rows), adopts returned lists."""

    def __init__(self, rng, home):
        self.rng, self.home, self.view, self.n = rng, home, [], 0

    def add(self, row):
        if row["role"] == "user" and self.view and self.view[-1]["role"] == "user":
            self.view[-1] = {**self.view[-1], "content": f"{self.view[-1]['content']}\n\n{row['content']}"}
        else:
            self.view.append(row)

    def exchange(self):
        rng, self.n = self.rng, self.n + 1
        n = self.n
        text = f"q{n}" + WORD * rng.randrange(1, 8)
        if rng.random() < 0.15:
            text = rng.choice(NEAR) + " " + text
        self.add({"role": "user", "content": text, "timestamp": 1_700_000_000.0 + n})
        for k in range(rng.randrange(0, 3)):
            cid = f"c{n}_{k}"
            kind = rng.randrange(6)
            args = json.dumps({"path": f"/w/f{n}.py", "flag": rng.choice([True, 1, 1.0, None])})
            if kind == 0:
                long = base64.b64encode(bytes((i * 37 + n) % 256 for i in range(3420))).decode("ascii")
                args = json.dumps({"content": "hdr\n" + "\n".join(long[i:i + 76] for i in range(0, len(long), 76))})
            self.add({"role": "assistant", "content": f"a{n}_{k}", "tool_calls": [
                {"id": cid, "type": "function", "function": {"name": "read_file", "arguments": args}}]})
            out = f"r{n}_{k}" + WORD * rng.choice([5, 40, 400])
            if kind == 1:
                out = _persisted(self.home, cid, "full spill " + out)
            # tests/harness/messages.py tool(): the gateway sends both name and tool_name on a tool row
            self.add({"role": "tool", "tool_call_id": cid, "name": "read_file", "tool_name": "read_file",
                      "content": out})
        self.add({"role": "assistant", "content": f"answer {n}" + WORD * rng.choice([1, 3, 6, 150])})

    def impure(self, engine):
        rng, tools = self.rng, [i for i, r in enumerate(self.view) if r["role"] == "tool"]
        choice = rng.randrange(5)
        if choice == 0 and len(tools) > 2:  # host salvage: an old output (> 200 chars) becomes the placeholder
            i = rng.choice(tools[:-2])
            if len(str(self.view[i]["content"])) > 200:
                self.view[i] = {**self.view[i], "content": PRUNED}
        elif choice == 1 and tools:  # an in-place steer on the newest tool row
            i = tools[-1]
            self.view[i] = {**self.view[i], "content": str(self.view[i]["content"]) + OOB}
        elif choice == 2:  # the host replays LCM's survival projection of a long stored row
            rows = {r["content"]: r["store_id"] for r in engine._store.get_session_messages(engine._session_id)
                    if isinstance(r.get("content"), str) and len(r["content"]) > 2400}
            for i, row in enumerate(self.view):
                if row.get("content") in rows and not row.get("timestamp"):
                    self.view[i] = engine._survival_projection([row], {id(row): rows[row["content"]]}, 0)[0]
                    break
        elif choice in (3, 4):  # a host that reloaded its transcript replays stored forms: externalized outputs and
            # ingest payload placeholders, as the store holds them
            stored = {}
            for r in engine._store.get_session_messages(engine._session_id):
                calls = r.get("tool_calls")
                calls = json.loads(calls) if isinstance(calls, str) else calls
                if r["role"] == "tool" and str(r.get("content")).startswith("[Externalized tool output"):
                    stored[("tool", r.get("tool_call_id"))] = {"content": r["content"]}
                elif calls and "[Externalized LCM ingest payload:" in json.dumps(calls):
                    stored[("assistant", calls[0].get("id"))] = {"tool_calls": calls}
            for i, row in enumerate(self.view):
                key = (row["role"], row.get("tool_call_id") if row["role"] == "tool" else
                       (row.get("tool_calls") or [{}])[0].get("id"))
                if key in stored and rng.random() < 0.5:
                    self.view[i] = {**row, **stored[key]}


def _dump_db(run):
    conn = sqlite3.connect(f"file:{run / 'lcm.db'}?mode=ro", uri=True)
    try:
        tables = [t for (t,) in conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
        out = {}
        for table in tables:
            try:
                out[table] = [list(r) for r in conn.execute(f'SELECT * FROM "{table}" ORDER BY rowid')]
            except sqlite3.OperationalError:  # WITHOUT ROWID or virtual tables
                out[table] = sorted([list(r) for r in conn.execute(f'SELECT * FROM "{table}"')], key=repr)
        # The one value no patch reaches: db_bootstrap stamps completed_at with SQLite's strftime('%s','now').
        out["lcm_migration_state"] = [[row[0], "<sql now>", *row[2:]] for row in out.get("lcm_migration_state", [])]
        # The FTS integrity scan's bookkeeping stamps: its claim takes a short-timeout write lock, so whether a
        # scan runs at a restart depends on lock timing (seen base-vs-base); the stamps are left out.
        out["metadata"] = [[row[0], _untimed_json(row[1]), *row[2:]] for row in out.get("metadata", [])
                           if not str(row[0]).startswith("fts_integrity_")]
        return out
    finally:
        conn.close()


def _ctime_free(value):
    """The spillover file's inode ctime (recorded in persisted-output metadata) is the one filesystem clock no
    patch can set; it is replaced by a token. The mtime is fixed when the file is written."""
    if isinstance(value, str):
        return CTIME.sub(r"\1<fs ctime>", value)
    if isinstance(value, dict):
        return {k: _ctime_free(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_ctime_free(v) for v in value)
    return value


def _untimed_json(text):
    """A metadata value without measured durations (``*_ms``, e.g. compaction telemetry's last duration)."""
    try:
        value = json.loads(text)
    except (TypeError, ValueError):
        return text
    return json.dumps(_untimed(value), sort_keys=True) if isinstance(value, dict) else text


def _untimed(value):
    """The status without its measured lock-wait durations (``*_ms``), the only wall-time values it reports, and
    without ``runtime_identity`` (each tree's own path, git branch and commit)."""
    if isinstance(value, dict):
        return {k: _untimed(v) for k, v in value.items() if not str(k).endswith("_ms") and k != "runtime_identity"}
    return value


def _files(run):
    return sorted((str(p.relative_to(run)), p.read_bytes().decode("utf-8", "replace"))
                  for p in run.rglob("*") if p.is_file() and not p.name.startswith("lcm.db"))


def _digest(row):
    """A stored row's content identity: sha256 of role and content. Copies are compared by this, never by id."""
    content = row.get("content")
    text = content if isinstance(content, str) else json.dumps(content, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(f"{row.get('role')}\x00{text}".encode("utf-8", "surrogatepass")).hexdigest()


def _classify(engine, returned, session, externalize):
    """One pass's stored rows of the session that are not COVERED. A row is covered when a byte-equal stored copy
    (same digest) is under a node whose summary is in ``returned``, or when its content (or, for an externalized
    row, the payload content it names) is in ``returned``: a plain row, or the text carried behind a summary head
    (never a summary body). An
    uncovered row whose digest repeats an earlier stored row is that row's copy and is judged once, on the earlier
    row. The rest is either behind a pointer (shown only as the host's persisted-output marker, the pruned
    placeholder or a survival projection of it) or a loss (class d). Entries carry role, digest and form only."""
    texts = [r.get("content") if isinstance(r.get("content"), str) else json.dumps(r.get("content"), sort_keys=True,
             ensure_ascii=False) for r in returned]
    joined = "\n".join(t or "" for t in texts)
    shown = {int(m.group(2)) for text in texts for m in SUMMARY_PART.finditer(text or "")}
    # Where a stored row's text may appear: a plain returned row whole, but a summary row only in the text carried
    # behind its summary parts. Text inside a summary body (a fallback summary can quote rows verbatim) is the
    # node's summary, not the row: the row counts as summarised only when it is under that node.
    carried = [t.rsplit(SUMMARY_EXPAND, 1)[-1].split("]", 1)[-1] if SUMMARY_PART.match(t or "") else t for t in texts]
    under, stack = set(), [n for n in map(engine._dag.get_node, sorted(shown)) if n is not None and n.summary in joined]
    while stack:
        node = stack.pop()
        if node.source_type == "messages":
            under.update(node.source_ids)
        else:
            stack.extend(n for n in map(engine._dag.get_node, node.source_ids) if n is not None)
    rows = engine._store.get_session_messages(session, limit=10 ** 7)
    covered = {_digest(r) for r in rows if r["store_id"] in under}
    # A survival projection points at the store row its mark names; under the copies rule it points at every
    # byte-equal stored copy of that row too.
    named = {int(m.group(1)) for t in texts for m in PROJECTION_NAMES.finditer(t or "")}
    projected = {_digest(r) for r in rows if r["store_id"] in named}
    seen, pointer, lost = set(), [], []
    for row in rows:
        digest, content = _digest(row), row.get("content") or ""
        first = digest not in seen
        seen.add(digest)
        if digest in covered:
            continue
        forms = [content]
        ref = externalize.extract_externalized_ref(content)
        payload = externalize.load_externalized_payload(ref, config=engine._config, hermes_home=engine._hermes_home) \
            if ref else None
        if payload is not None and isinstance(payload.get("content"), str):
            forms.append(payload["content"])
        if any(form == t or (form and t and form in t) for form in forms for t in carried):
            covered.add(digest)
            continue
        if not first:
            continue  # a copy of an earlier stored row: judged on that row
        shown_as = sorted({"pruned" if r.get("content") == PRUNED else "persisted_marker"
                           for r in returned if row.get("tool_call_id") and r.get("tool_call_id") == row["tool_call_id"]
                           and (r.get("content") == PRUNED or str(r.get("content")).lstrip().startswith("<persisted-output>"))})
        if digest in projected:
            shown_as.append("projection")
        entry = (row["role"], digest[:16], "externalized" if ref else "inline", content[:60])
        (pointer if shown_as else lost).append(entry + (shown_as,) if shown_as else entry)
    return pointer, lost


def _stored_again(rows, ranges):
    """Stored rows of the session whose digest equals an earlier stored row's, by the op whose store-id range
    (lowest id above the previous op's highest, through this op's highest) wrote them."""
    seen, out = set(), Counter()
    for row in rows:
        digest = _digest(row)
        if digest in seen:
            out[next((op for op, low, high in ranges if low < row["store_id"] <= high), "unknown")] += 1
        seen.add(digest)
    return dict(out)


def _run(package, run, seed, clock):
    rng = random.Random(seed)
    engine_mod, config_mod = (importlib.import_module(f"{package.__name__}.{m}") for m in ("engine", "config"))
    importlib.import_module(f"{package.__name__}.escalation")._call_llm_for_summary = _summariser
    externalize = importlib.import_module(f"{package.__name__}.externalize")
    # The status's git identity runs `git status` in the tree with a 1 s timeout; a killed probe leaves the
    # tree's index.lock behind. The identity is left out of the dump anyway, so it is not probed.
    engine_mod._git_runtime_identity = lambda root: dict(plugin_git_commit="", plugin_git_branch="",
                                                         plugin_git_dirty=None, plugin_git_remote="")
    bootstrap = importlib.import_module(f"{package.__name__}.db_bootstrap")

    def settle():  # the store's background FTS integrity scan and stem backfill race any write and the dump
        bootstrap.join_background_integrity_scans()
        bootstrap.join_background_stem_backfills()
    home = run / "home"
    config = dict(database_path=str(run / "lcm.db"), fresh_tail_count=6, large_output_externalization_enabled=True,
                  large_output_externalization_threshold_chars=3000, large_output_externalization_path=str(run / "ext"),
                  # Off: the rollup maintenance and embedding threads write the store on their own schedule (a
                  # race, base-vs-base too); neither reads a replay identity.
                  temporal_rollups_enabled=False, embeddings_enabled=False,
                  leaf_chunk_tokens=800)  # small enough that these short sessions make summaries

    def open_engine():
        engine = engine_mod.LCMEngine(config=config_mod.LCMConfig(**config), hermes_home=str(home))
        engine.on_session_start("S", platform="cli", context_length=200_000, conversation_id="conv")
        settle()
        return engine

    steps, host, kinds = [], Host(rng, home), Counter()
    pointer, lost, ranges, high = [], [], [], 0

    def ingest(view):
        for row in view:
            text = json.dumps(row.get("content"), ensure_ascii=False) + json.dumps(row.get("tool_calls"))
            kinds.update(kind for kind, marker in KIND_MARKERS if marker in text)
            kinds["pure"] += not any(marker in text for _kind, marker in KIND_MARKERS)
            kinds["carrier"] += bool(row.get("role") == "user" and SUMMARY_PART.match(str(row.get("content")))
                                     and "\n\n" in str(row.get("content")).split("[Expand for details:", 1)[-1])
        engine.ingest(deepcopy(view))
    engine = open_engine()
    plan = ["turn"] * 14 + ["impure"] * 5 + ["pass"] * 4 + ["rebind", "switch", "restart"]
    rng.shuffle(plan)
    try:
        for index, op in enumerate(["turn", "turn"] + plan + ["pass", "end"]):
            clock[0] = 1_700_000_000.0 + 60 * index
            settle()
            returned = None
            if op == "turn":
                host.exchange()
                engine.should_compress_preflight(deepcopy(host.view))
                ingest(host.view)
            elif op == "impure":
                host.impure(engine)
                ingest(host.view)
            elif op == "pass":
                returned = engine.compress(deepcopy(host.view), force=True)
                behind, missing = _classify(engine, returned, "S", externalize)
                pointer += [(index, *entry) for entry in behind]
                lost += [(index, *entry) for entry in missing]
                host.view = []
                for row in deepcopy(returned):  # the host's alternation repair glues a user row to the summary
                    host.add(row)
            elif op == "rebind":
                engine.on_session_start("S", platform="cli", context_length=200_000, conversation_id="conv")
            elif op == "switch":
                engine.on_session_start("S2", platform="cli", context_length=200_000, conversation_id="conv2")
                settle()
                engine.ingest([{"role": "user", "content": "side question"}, {"role": "assistant", "content": "ok"}])
                engine.on_session_start("S", platform="cli", context_length=200_000, conversation_id="conv")
            elif op == "restart":
                engine.shutdown()
                engine = open_engine()
            else:
                engine.on_session_end("S", deepcopy(host.view))
            settle()
            status = _untimed(engine.get_status())
            stored = [r for r in engine._store.get_session_messages("S", limit=10 ** 7)]
            low, high = high, max([high, *(r["store_id"] for r in stored)])
            ranges.append((op, low, high))
            steps.append(_ctime_free(dict(op=op, cursor=getattr(engine, "_ingest_cursor", None), returned=returned,
                              status=json.loads(json.dumps(status, default=repr, sort_keys=True)),
                              db=_dump_db(run), files=_files(run))))
    finally:
        engine.shutdown()
    again = _stored_again(stored, ranges)
    return steps, dict(pointer=pointer, lost=lost, stored_again=again), kinds


def _session(name, tree, seed, root, monkeypatch, rep=0):
    run = root / "run"
    run.mkdir()
    clock = [0.0]
    entropy = random.Random(f"entropy-{seed}")  # the store's salt, token and uuid, equal in both trees
    with monkeypatch.context() as patch:
        gmtime, localtime = time.gmtime, time.localtime
        patch.setattr(time, "time", lambda: clock[0])
        patch.setattr(time, "time_ns", lambda: int(clock[0] * 1e9))
        patch.setattr(time, "gmtime", lambda secs=None: gmtime(clock[0] if secs is None else secs))
        patch.setattr(time, "localtime", lambda secs=None: localtime(clock[0] if secs is None else secs))
        patch.setattr(secrets, "token_bytes", lambda n=32: entropy.randbytes(n))
        patch.setattr(secrets, "token_hex", lambda n=32: entropy.randbytes(n).hex())
        patch.setattr(uuid, "uuid4", lambda: uuid.UUID(int=entropy.getrandbits(128), version=4))
        patch.setattr(tempfile, "tempdir", str(run))
        patch.setenv("HERMES_HOME", str(run / "home"))
        patch.setenv("LCM_EMBEDDINGS_ENABLED", "0")
        patch.setenv("LCM_BACKGROUND_LEAF_PREP_ENABLED", "0")
        patch.delenv("LCM_HOST_MESSAGE_UID", raising=False)
        result = _run(_load(tree, name), run, seed, clock)
    os.rename(run, root / f"{name}-{seed}-{rep}")  # keep it, and free the path for the next tree
    return result


def _paths(left, right, path=""):
    if isinstance(left, dict) and isinstance(right, dict):
        return [p for k in sorted(set(left) | set(right), key=str) for p in _paths(left.get(k), right.get(k), f"{path}.{k}")]
    return [] if left == right else [(path, left, right)]


def _first_difference(left, right):
    for index, (a, b) in enumerate(zip(left, right)):
        if a != b:
            keys = [k for k in a if a[k] != b.get(k)]
            detail = {k: (a[k], b[k]) for k in keys if k not in ("db", "files", "status")}
            if "status" in keys:
                detail["status"] = _paths(a["status"], b["status"])
            if "files" in keys:
                left_files, right_files = dict(a["files"]), dict(b["files"])
                detail["files"] = sorted(f for f in set(left_files) | set(right_files)
                                         if left_files.get(f) != right_files.get(f))
            if "db" in keys:
                detail["db"] = {t: ([r for r in a["db"][t] if r not in b["db"].get(t, [])][:3],
                                    [r for r in b["db"].get(t, []) if r not in a["db"][t]][:3])
                                for t in a["db"] if a["db"][t] != b["db"].get(t)}
            return index, a["op"], detail
    return None if len(left) == len(right) else ("length", len(left), len(right))


def test_a6_harness_is_deterministic(tmp_path, monkeypatch):
    for seed in (0, 1):
        first = _session("_fw_det", CANDIDATE, seed, tmp_path, monkeypatch, rep=0)[0]
        second = _session("_fw_det", CANDIDATE, seed, tmp_path, monkeypatch, rep=1)[0]
        assert _first_difference(first, second) is None, _first_difference(first, second)


_RUNS = {}
# Seeds whose BASE run has rows neither summarised nor shown (class d; #2890 on main), 30 seeds.
BASE_LOSS_SEEDS = frozenset({1, 3, 5, 7, 12, 16, 18, 20, 21, 22, 25, 26, 29})
LOSS_REASON = "rows neither summarised nor shown on the base: under investigation (#2856 differential)"


class RowsNeitherSummarisedNorShown(AssertionError):
    """Raised only for class (d) rows, so no other failure can satisfy the expected failure."""


def _pair(seed, tmp_path_factory, monkeypatch):
    """Both trees' runs of one seed, made once and shared by the tests below."""
    if seed not in _RUNS:
        root = tmp_path_factory.mktemp(f"a6-seed-{seed}")
        _RUNS[seed] = (_session("_fw_base", Path(BASE), seed, root, monkeypatch),
                       _session("_fw_candidate", CANDIDATE, seed, root, monkeypatch))
        (base, base_rows, kinds), (candidate, candidate_rows, _) = _RUNS[seed]
        out = os.environ.get("LCMX_FIXEDWORK_EVIDENCE")
        if out:
            Path(out, "a6-hostfaithful").mkdir(exist_ok=True)
            Path(out, "a6-hostfaithful", f"seed-{seed}.json").write_text(json.dumps(dict(
                seed=seed, steps=len(base), ops=[s["op"] for s in base], passes=sum(s["op"] == "pass" for s in base),
                stored_rows=len(base[-1]["db"].get("messages", [])),
                summary_nodes=len(base[-1]["db"].get("summary_nodes", [])), ingested_kinds=dict(kinds),
                equal=_first_difference(base, candidate) is None,
                base=base_rows, candidate=candidate_rows), indent=1, default=repr) + "\n")
    return _RUNS[seed]


@pytest.mark.parametrize("seed", range(SEEDS))
def test_a6_candidate_matches_base_per_step(tmp_path_factory, monkeypatch, seed):
    (base, _, _), (candidate, _, _) = _pair(seed, tmp_path_factory, monkeypatch)
    assert _first_difference(base, candidate) is None, _first_difference(base, candidate)


@pytest.mark.parametrize("seed", [pytest.param(seed, marks=pytest.mark.xfail(
    strict=True, raises=RowsNeitherSummarisedNorShown, reason=LOSS_REASON)) if seed in BASE_LOSS_SEEDS else seed
    for seed in range(SEEDS)])
def test_a6_every_pass_keeps_every_unsummarised_row(tmp_path_factory, monkeypatch, seed):
    """On both trees: rows stored again and rows behind a pointer are counted, not losses; the candidate never
    puts more rows behind a pointer than the base and loses exactly the base's rows. Any row left (class d) fails
    the seed; a seed whose base has such rows is an expected failure until it is explained."""
    (_, base, _), (_, candidate, _) = _pair(seed, tmp_path_factory, monkeypatch)
    assert len(candidate["pointer"]) <= len(base["pointer"]), (seed, base["pointer"], candidate["pointer"])
    assert candidate["lost"] == base["lost"], (seed, base["lost"], candidate["lost"])
    if base["lost"]:
        raise RowsNeitherSummarisedNorShown((seed, base["lost"]))


def test_a6_sessions_exercise_every_kind(tmp_path_factory, monkeypatch):
    kinds, nodes, lines = Counter(), 0, []
    totals = {tree: dict(again=Counter(), pointer=0, lost=0) for tree in ("base", "candidate")}
    for seed in range(SEEDS):
        (base, base_rows, seed_kinds), (_, candidate_rows, _) = _pair(seed, tmp_path_factory, monkeypatch)
        kinds.update(seed_kinds)
        nodes += len(base[-1]["db"].get("summary_nodes", []))
        for tree, rows in (("base", base_rows), ("candidate", candidate_rows)):
            totals[tree]["again"].update(rows["stored_again"])
            totals[tree]["pointer"] += len(rows["pointer"])
            totals[tree]["lost"] += len(rows["lost"])
        lines.append(f"seed {seed}: stored again base {base_rows['stored_again']} candidate "
                     f"{candidate_rows['stored_again']}; behind a pointer base {len(base_rows['pointer'])} candidate "
                     f"{len(candidate_rows['pointer'])}; lost (d) base {len(base_rows['lost'])} candidate "
                     f"{len(candidate_rows['lost'])}")
    out = os.environ.get("LCMX_FIXEDWORK_EVIDENCE")
    if out:
        Path(out, "a6-hostfaithful").mkdir(exist_ok=True)
        Path(out, "a6-hostfaithful", "kinds.json").write_text(json.dumps(dict(kinds=dict(kinds), summary_nodes=nodes),
                                                                   indent=1) + "\n")
        Path(out, "a6-hostfaithful", "a6_hostfaithful_summary.txt").write_text("\n".join(lines + [
            f"total {tree}: stored again {dict(t['again'])} (all {sum(t['again'].values())}); behind a pointer "
            f"{t['pointer']}; lost (d) {t['lost']}" for tree, t in totals.items()]) + "\n")
    assert nodes > 0
    assert all(kinds[kind] > 0 for kind, _ in KIND_MARKERS), kinds
    assert kinds["carrier"] > 0 and kinds["pure"] > 0, kinds
