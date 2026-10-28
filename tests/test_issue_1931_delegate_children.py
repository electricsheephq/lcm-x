"""#1931: delegated subagent sessions are stored under their own id with a parent link, never managed.

The synthetic repro from the issue: a delegated child's tool result carries a marker that exists nowhere
else. The child's rows must land under the child's session id (own conversation id), the parent must find
the marker through ``lcm_grep`` scope ``current`` and ``lcm_recall`` with the lineage shown, and nothing
of the parent may change. AUXILIARY sessions keep storing nothing.
"""

import importlib
import json
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from tests.test_issue_622_slow_load import _Ctx, _Manager, _install_host, _load_plugin

MARKER = "quokkaledger"
DELEGATE = {"_subagent_id": "sa-1", "_delegate_depth": 1, "log_prefix": "[subagent-0]",
            "enabled_toolsets": ["terminal", "file"]}


class Agent:
    """The host agent frame: hooks and engine calls run inside its methods, as in Hermes."""

    def __init__(self, session_id, parent="parent", **markers):
        self.session_id = session_id
        self._parent_session_id = parent
        for name, value in markers.items():
            setattr(self, name, value)

    def start(self, engine):
        engine.on_session_start(self.session_id, platform="subagent", context_length=200_000)

    def turn(self, ctx, history, engine=None):
        extra = {} if engine is None else {"context_compressor": engine}
        ctx.hooks["post_llm_call"][0](session_id=self.session_id, platform="subagent",
                                      conversation_history=list(history), **extra)

    def end(self, engine, history):
        engine.on_session_end(self.session_id, list(history))

    def compress(self, engine, history):
        return engine.compress(history, current_tokens=1_000_000, force=True)

    def rotate(self, engine, session_id):
        old = self.session_id
        self.session_id = session_id
        engine.on_session_start(session_id, platform="subagent", boundary_reason="compression", old_session_id=old)


def child_view(n_final=0):
    rows = [{"role": "user", "content": "Inspect the ledger and report."},
            {"role": "assistant", "content": "Checking the ledger.", "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "terminal", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "c1", "content": f"ledger line: {MARKER} balance 42"}]
    return rows + [{"role": "assistant", "content": f"final answer {i}"} for i in range(n_final)]


PARENT_VIEW = [{"role": "user", "content": "Delegate the ledger check."},
               {"role": "assistant", "content": "The subagent reported a balance of 42."}]


@pytest.fixture
def host(tmp_path, monkeypatch):
    monkeypatch.setenv("LCM_EMBEDDINGS_ENABLED", "false")
    _install_host(monkeypatch, tmp_path)
    opened, names = [], iter(f"hermes_lcm_1931_{id(tmp_path)}_{i}" for i in range(10))

    def load():
        name = next(names)
        module = _load_plugin(name)
        ctx = _Ctx(_Manager())
        module.register(ctx)
        opened.append(ctx.offered)
        return ctx, (lambda sub: importlib.import_module(f"{name}.{sub}"))

    def clone(ctx):
        engine = ctx.offered.clone_for_agent()
        opened.append(engine)
        return engine

    yield load, clone
    for engine in reversed(opened):
        engine.shutdown()


def parent_state(engine):
    conv = engine._conversation_id
    conn = engine._store.connection
    relations = conn.execute("SELECT COUNT(*) FROM message_relations").fetchone()[0] if conn.execute(
        "SELECT 1 FROM sqlite_master WHERE name='message_relations'").fetchone() else 0
    state = engine._lifecycle.get_by_conversation(conv)
    return (engine._session_id, conv, engine._ingest_cursor, engine.current_session_id,
            [(m["store_id"], m["role"], m["content"]) for m in engine._store.get_session_messages("parent")],
            len(engine._dag.get_session_nodes("parent")), relations,
            (state.current_session_id, state.current_frontier_store_id, state.last_finalized_session_id))


def child_rows(engine, session_id="child"):
    return engine._store.get_session_messages(session_id)


def lineage_rows(engine):
    conn = engine._store.connection
    if not conn.execute("SELECT 1 FROM sqlite_master WHERE name='session_lineage'").fetchone():
        return []
    return conn.execute("SELECT child_session_id, parent_session_id, source FROM session_lineage").fetchall()


@pytest.mark.parametrize("route", ["shared", "clone", "resident_no_engine", "no_resident"])
def test_synthetic_repro_flips_child_rows_stored_and_recallable(host, route):
    load, clone = host
    ctx, module = load()
    child = Agent("child", **DELEGATE)
    parent = None
    if route != "no_resident":
        parent = clone(ctx)
        parent.on_session_start("parent", platform="cli", context_length=200_000)
        ctx.hooks["post_llm_call"][0](session_id="parent", platform="cli",
                                      conversation_history=PARENT_VIEW, context_compressor=parent)
        before = parent_state(parent)
    engine = {"shared": parent, "resident_no_engine": None, "no_resident": None}.get(route)
    if route == "clone":
        engine = clone(ctx)
        child.start(engine)
    for n in (0, 1, 2):
        child.turn(ctx, child_view(n), engine)
    if parent is None:
        parent = clone(ctx)
        parent.on_session_start("parent", platform="cli", context_length=200_000)
    else:
        assert parent_state(parent) == before  # isolation: nothing of the parent moved
    rows = child_rows(parent)
    assert [m["content"] for m in rows] == [m["content"] for m in child_view(2)]
    assert {m["conversation_id"] for m in rows} == {"child"}
    assert lineage_rows(parent) == [("child", "parent", "parent_frame")]
    assert not any(MARKER in str(m["content"]) for m in parent._store.get_session_messages("parent"))

    grep = json.loads(module("tools").lcm_grep({"query": MARKER}, engine=parent))
    hits = [hit for hit in grep["results"] if hit.get("session_id") == "child"]
    assert hits and hits[0]["lineage"] == "subagent of parent" and hits[0]["parent_session_id"] == "parent"
    recall = json.loads(module("tools").lcm_recall({"query": MARKER}, engine=parent))
    hits = [hit for hit in recall["hits"] if hit.get("session_id") == "child"]
    assert hits and hits[0]["lineage"] == "subagent of parent"


def test_exactly_once_resend_retry_resume_and_host_rewrite(host):
    load, clone = host
    ctx, _ = load()
    parent = clone(ctx)
    parent.on_session_start("parent", platform="cli", context_length=200_000)
    child = Agent("child", **DELEGATE)
    child.turn(ctx, child_view(0), parent)
    child.turn(ctx, child_view(0), parent)  # identical view re-sent / hook retried
    assert len(child_rows(parent)) == 3
    child.turn(ctx, child_view(1), parent)  # one more row stores only that row
    assert len(child_rows(parent)) == 4
    child.end(parent, child_view(1))  # session end replays the same view
    assert len(child_rows(parent)) == 4
    # Resumed in a new process: the replayed history plus one new row.
    ctx2, _ = load()
    parent2 = clone(ctx2)
    parent2.on_session_start("parent", platform="cli", context_length=200_000)
    child.turn(ctx2, child_view(2), parent2)
    assert [m["content"] for m in child_rows(parent2)] == [m["content"] for m in child_view(2)]
    # The host's native compaction rewrote the head: resume after the unique stored tail.
    rewritten = [{"role": "user", "content": "[CONTEXT COMPACTION] earlier turns summarised"},
                 *child_view(2)[-3:], {"role": "assistant", "content": "after compaction"}]
    child.turn(ctx2, rewritten, parent2)
    child.turn(ctx2, rewritten, parent2)
    contents = [m["content"] for m in child_rows(parent2)]
    assert contents == [m["content"] for m in child_view(2)] + ["after compaction"]


def test_unmatched_child_view_is_a_visible_fallback(host, caplog):
    load, clone = host
    ctx, module = load()
    parent = clone(ctx)
    parent.on_session_start("parent", platform="cli", context_length=200_000)
    child = Agent("child", **DELEGATE)
    child.turn(ctx, child_view(0), parent)
    before = module("aux_session").delegate_child_store_counts().get("fallbacks", 0)
    with caplog.at_level("WARNING"):
        child.turn(ctx, [{"role": "user", "content": "an unrelated history"}], parent)
    assert [row["content"] for row in child_rows(parent)] == [row["content"] for row in child_view()] + ["an unrelated history"]
    assert module("aux_session").delegate_child_store_counts()["fallbacks"] == before + 1
    assert any("delegated child child" in r.getMessage() and "counted identities" in r.getMessage()
               for r in caplog.records)


@pytest.mark.parametrize("markers", [
    {**DELEGATE, "_memory_write_origin": "background_review"},
    {**DELEGATE, "_memory_write_context": "background_review"},
    {**DELEGATE, "enabled_toolsets": ["memory", "skills"]},
], ids=["review-origin", "review-context", "memory-skills-only"])
@pytest.mark.parametrize("route", ["shared", "resident_no_engine"])
def test_auxiliary_sessions_store_nothing(host, markers, route):
    load, clone = host
    ctx, _ = load()
    parent = clone(ctx)
    parent.on_session_start("parent", platform="cli", context_length=200_000)
    before = parent_state(parent)
    aux = Agent("aux", **markers)
    for n in (0, 1):
        aux.turn(ctx, child_view(n), parent if route == "shared" else None)
    if route == "shared":
        aux.end(parent, child_view(1))
    assert child_rows(parent, "aux") == []
    assert lineage_rows(parent) == []
    assert parent_state(parent) == before


def test_auxiliary_call_without_delegate_signal_stores_nothing(host):
    load, clone = host
    ctx, _ = load()
    parent = clone(ctx)
    parent.on_session_start("parent", platform="cli", context_length=200_000)
    Agent("title-call", parent="parent").turn(ctx, child_view(0))  # no delegate marker, record or state.db row
    assert child_rows(parent, "title-call") == []
    assert lineage_rows(parent) == []


def test_explicit_hook_and_state_db_signals_name_their_source(host, tmp_path):
    load, clone = host
    ctx, module = load()
    parent = clone(ctx)
    parent.on_session_start("parent", platform="cli", context_length=200_000)
    ctx.hooks["subagent_start"][0](child_session_id="hooked", parent_session_id="parent",
                                   child_subagent_id="sa-9", child_role="leaf")
    Agent("hooked", parent="").turn(ctx, child_view(0))
    with sqlite3.connect(tmp_path / "state.db") as conn:
        conn.execute("CREATE TABLE sessions (id TEXT PRIMARY KEY, source TEXT NOT NULL, parent_session_id TEXT)")
        conn.execute("INSERT INTO sessions VALUES ('hosted', 'subagent', 'parent'), ('plain', 'cli', 'parent')")
    Agent("hosted", parent="").turn(ctx, child_view(0))
    Agent("plain", parent="").turn(ctx, child_view(0))
    assert sorted(lineage_rows(parent)) == [("hooked", "parent", "subagent_start"), ("hosted", "parent", "state_db")]
    assert len(child_rows(parent, "hooked")) == len(child_rows(parent, "hosted")) == 3
    assert child_rows(parent, "plain") == []
    row = parent._store.connection.execute(
        "SELECT child_subagent_id, child_role FROM session_lineage WHERE child_session_id='hooked'").fetchone()
    assert tuple(row) == ("sa-9", "leaf")


def test_recall_scopes_for_child_rows(host):
    load, clone = host
    ctx, module = load()
    parent = clone(ctx)
    parent.on_session_start("parent", platform="cli", context_length=200_000)
    Agent("child", **DELEGATE).turn(ctx, child_view(0), parent)
    Agent("grandchild", parent="child", **DELEGATE).turn(ctx, child_view(0))
    Agent("sibling-child", parent="sibling", **DELEGATE).turn(ctx, child_view(0))  # another parent's child
    grep = module("tools").lcm_grep

    def sessions(**args):
        payload = json.loads(grep({"query": MARKER, **args}, engine=parent))
        return {hit["session_id"]: hit.get("lineage") for hit in payload["results"]}

    assert sessions() == {"child": "subagent of parent", "grandchild": "subagent of child"}
    assert sessions(session_scope="all") == {"child": "subagent of parent", "grandchild": "subagent of child",
                                             "sibling-child": "subagent of sibling"}
    assert sessions(session_scope="session", session_id="sibling-child") == {"sibling-child": "subagent of sibling"}
    # Deleting a parent's messages keeps its children's links; deleting a child's removes the child's own link.
    parent._store.delete_session_messages("parent")
    parent._store.delete_session_messages("grandchild")
    assert sorted(r[0] for r in lineage_rows(parent)) == ["child", "sibling-child"]
    assert sessions(session_scope="session", session_id="child") == {"child": "subagent of parent"}


@pytest.mark.parametrize("pattern_kind", ["IGNORE", "STATELESS"])
@pytest.mark.parametrize("target", ["bound-parent", "recorded-parent", "host-parent", "child"])
def test_platform_patterns_exclude_delegate_storage(host, tmp_path, monkeypatch, pattern_kind, target):
    monkeypatch.setenv(f"LCM_{pattern_kind}_SESSION_PATTERNS", "subagent" if target == "child" else "cron:*")
    load, clone = host
    ctx, _ = load()
    parent = clone(ctx)
    parent.on_session_start("parent", platform="cron", context_length=200_000)
    if target == "recorded-parent":
        parent.on_session_start("foreground", platform="cli", context_length=200_000)
        ctx.hooks["subagent_start"][0](child_session_id="child", parent_session_id="parent")
        parent._store_delegate_child_turn("child", child_view(), platform="subagent")
    elif target == "host-parent":
        with sqlite3.connect(tmp_path / "state.db") as conn:
            conn.execute("CREATE TABLE sessions (id TEXT PRIMARY KEY, source TEXT, parent_session_id TEXT)")
            conn.execute("INSERT INTO sessions VALUES ('parent', 'cron', NULL)")
        other = clone(ctx)
        child = Agent("child", **DELEGATE)
        child.start(other)
        child.turn(ctx, child_view(), other)
    else:
        Agent("child", **DELEGATE).turn(ctx, child_view(), parent)
    assert child_rows(parent) == []
    assert lineage_rows(parent) == []


@pytest.mark.parametrize("markers", [
    {**DELEGATE, "_memory_write_origin": "background_review"},
    {**DELEGATE, "enabled_toolsets": ["memory", "skills"]},
], ids=["background-review", "memory-only"])
@pytest.mark.parametrize("seen_at", ["start", "turn"])
def test_auxiliary_classification_survives_frame_free_end(host, tmp_path, markers, seen_at):
    load, clone = host
    ctx, _ = load()
    parent = clone(ctx)
    parent.on_session_start("parent", platform="cli", context_length=200_000)
    ctx.hooks["subagent_start"][0](child_session_id="child", parent_session_id="parent")
    with sqlite3.connect(tmp_path / "state.db") as conn:
        conn.execute("CREATE TABLE sessions (id TEXT PRIMARY KEY, source TEXT, parent_session_id TEXT)")
        conn.execute("INSERT INTO sessions VALUES ('child', 'subagent', 'parent')")
    aux = Agent("child", **markers)
    if seen_at == "start":
        aux.start(parent)
    else:
        aux.turn(ctx, child_view(), parent)
    parent.on_session_end("child", child_view(1))  # no agent frame; both delegate signals remain
    assert child_rows(parent) == []
    assert lineage_rows(parent) == []


def test_lineage_write_retried_on_identical_view(host, monkeypatch):
    load, clone = host
    ctx, _ = load()
    parent = clone(ctx)
    parent.on_session_start("parent", platform="cli", context_length=200_000)
    original = parent._store.record_session_lineage
    attempts = []

    def fail_once(*args, **kwargs):
        attempts.append(args)
        if len(attempts) == 1:
            raise sqlite3.OperationalError("synthetic link failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(parent._store, "record_session_lineage", fail_once)
    child = Agent("child", **DELEGATE)
    child.turn(ctx, child_view(), parent)
    assert len(child_rows(parent)) == 3 and lineage_rows(parent) == []
    child.turn(ctx, child_view(), parent)
    assert len(child_rows(parent)) == 3
    assert lineage_rows(parent) == [("child", "parent", "parent_frame")]
    assert len(attempts) == 2


def test_stored_lineage_survives_stop_and_frame_free_end(host):
    load, clone = host
    ctx, _ = load()
    parent = clone(ctx)
    parent.on_session_start("parent", platform="cli", context_length=200_000)
    ctx.hooks["subagent_start"][0](child_session_id="child", parent_session_id="parent")
    Agent("child", parent="").turn(ctx, child_view())  # resident-conflict store point
    ctx.hooks["subagent_stop"][0](child_session_id="child")
    parent.on_session_end("child", child_view(1))
    assert [row["content"] for row in child_rows(parent)] == [row["content"] for row in child_view(1)]
    assert lineage_rows(parent) == [("child", "parent", "subagent_start")]


def test_delegate_without_parent_warns_and_counts(host, caplog):
    load, clone = host
    ctx, module = load()
    parent = clone(ctx)
    parent.on_session_start("parent", platform="cli", context_length=200_000)
    before = module("aux_session").delegate_child_store_counts().get("fallbacks", 0)
    with caplog.at_level("WARNING"):
        Agent("child", parent="", **DELEGATE).turn(ctx, child_view())
    assert child_rows(parent) == [] and lineage_rows(parent) == []
    assert module("aux_session").delegate_child_store_counts()["fallbacks"] == before + 1
    assert any("delegated child child" in record.message and "parent identity" in record.message
               for record in caplog.records)


def test_overlapping_store_points_across_clones_append_once(host, monkeypatch):
    load, clone = host
    ctx, module = load()
    first, second = clone(ctx), clone(ctx)
    for engine in (first, second):
        engine.on_session_start("parent", platform="cli", context_length=200_000)
    ctx.hooks["subagent_start"][0](child_session_id="child", parent_session_id="parent")
    entered, release = threading.Event(), threading.Event()
    aux = module("aux_session")
    lock = getattr(aux, "_DELEGATE_STORE_LOCK", threading.RLock())

    class ObservedLock:
        def __enter__(self):
            if entered.is_set():
                release.set()  # contender is at the lock while the first append is held
            lock.acquire()

        def __exit__(self, *args):
            lock.release()

    monkeypatch.setattr(aux, "_DELEGATE_STORE_LOCK", ObservedLock(), raising=False)
    append_first = first._append_off_current_session_end_suffix
    append_second = second._append_off_current_session_end_suffix

    def hold_first(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return append_first(*args, **kwargs)

    def racing_second(*args, **kwargs):
        release.set()  # without the fix, both suffixes have already been matched
        return append_second(*args, **kwargs)

    monkeypatch.setattr(first, "_append_off_current_session_end_suffix", hold_first)
    monkeypatch.setattr(second, "_append_off_current_session_end_suffix", racing_second)
    with ThreadPoolExecutor(max_workers=2) as pool:
        one = pool.submit(first._store_delegate_child_turn, "child", child_view())
        assert entered.wait(5)
        two = pool.submit(second._store_delegate_child_turn, "child", child_view())
        assert one.result(timeout=10) + two.result(timeout=10) == 3
    assert [row["content"] for row in child_rows(first)] == [row["content"] for row in child_view()]


def test_session_lineage_is_known_to_schema_stamp_doctor(host):
    load, clone = host
    ctx, module = load()
    engine = clone(ctx)
    bootstrap = module("db_bootstrap")
    # A complete core schema keeps this check about the one additive table.
    with engine._store.atomic():
        bootstrap.set_schema_version(engine._store.connection, bootstrap.SCHEMA_VERSION + 1)
    before = bootstrap.classify_version_mismatch(engine._store.connection)
    assert before == bootstrap.VERSION_MISMATCH_INTERIM_STAMP
    engine._store.record_session_lineage("child", "parent", source="subagent_start")
    assert bootstrap.classify_version_mismatch(engine._store.connection) == before


def test_grep_schema_current_names_delegated_descendants(host):
    load, _ = host
    _, module = load()
    schema = module("schemas").LCM_GREP
    description = schema["parameters"]["properties"]["session_scope"]["description"]
    assert "active session and its delegated descendants" in description


@pytest.mark.parametrize("route", ["shared", "clone"])
@pytest.mark.parametrize("rotate", [False, True], ids=["same-id", "rotated-id"])
def test_compress_bypass_stores_before_rewrite_and_resumes(host, monkeypatch, route, rotate):
    load, clone = host
    ctx, _ = load()
    parent = clone(ctx)
    parent.on_session_start("parent", platform="cli", context_length=200_000)
    before = parent_state(parent)
    engine = parent if route == "shared" else clone(ctx)
    child = Agent("child", **DELEGATE)
    child.start(engine)
    view = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"turn {i} " + "payload " * 400}
            for i in range(12)]
    monkeypatch.setattr(engine, "_get_host_fallback_compressor", lambda: None)
    monkeypatch.setattr(engine, "_bypass_compaction_target_tokens", lambda **kwargs: 64)
    compacted = child.compress(engine, view)  # actual bypass compressor, in the child's frame
    assert [row["content"] for row in child_rows(engine)] == [row["content"] for row in view]
    assert compacted != view and len(compacted) < 3
    if rotate:
        child.rotate(engine, "continued")
    for i in range(2):
        compacted = [*compacted, {"role": "assistant", "content": f"after compression {i}"}]
        child.turn(ctx, compacted, engine)
        child.turn(ctx, compacted, engine)
    child.end(engine, compacted)
    continued = child_rows(engine, "continued") if rotate else []
    assert [row["content"] for row in [*child_rows(engine), *continued]] == (
        [row["content"] for row in view] + ["after compression 0", "after compression 1"])
    if rotate:
        assert ("continued", "child", "compression") in lineage_rows(engine)
        assert set(engine._store.session_lineage_descendants("parent")) == {"child", "continued"}
        assert [row["content"] for row in continued] == ["after compression 0", "after compression 1"]
    assert parent_state(parent) == before


@pytest.mark.parametrize("kept", [1, 2])
def test_short_stored_tail_recovers_later_turns(host, kept):
    load, clone = host
    ctx, _ = load()
    parent = clone(ctx)
    parent.on_session_start("parent", platform="cli", context_length=200_000)
    child = Agent("child", **DELEGATE)
    original = child_view(3)
    child.turn(ctx, original, parent)
    rewritten = [{"role": "user", "content": "rewritten head"}, *original[-kept:]]
    for i in range(2):
        rewritten = [*rewritten, {"role": "assistant", "content": f"later {i}"}]
        child.turn(ctx, rewritten, parent)
        child.turn(ctx, rewritten, parent)
    assert [row["content"] for row in child_rows(parent)] == (
        [row["content"] for row in original] + ["later 0", "later 1"])


@pytest.mark.parametrize("shape", ["repeated", "rewritten"])
def test_unanchored_multiset_preserves_new_occurrences(host, shape, caplog):
    load, clone = host
    ctx, module = load()
    parent = clone(ctx)
    parent.on_session_start("parent", platform="cli", context_length=200_000)
    child = Agent("child", **DELEGATE)
    original = [{"role": "user", "content": "old head"},
                {"role": "assistant", "content": "repeat"},
                {"role": "user", "content": "repeat"},
                {"role": "assistant", "content": "repeat"}]
    child.turn(ctx, original, parent)
    if shape == "repeated":
        view = original[-3:] * 2
        unseen = original[-3:]
    else:
        view = [{**row, "content": "rewritten " + row["content"]} for row in original]
        unseen = view[:]
    view = [*view, {"role": "user", "content": "new turn"}]
    before = module("aux_session").delegate_child_store_counts().get("fallbacks", 0)
    with caplog.at_level("WARNING"):
        child.turn(ctx, view, parent)
        child.turn(ctx, view, parent)
    expected = [*original, *unseen, view[-1]]
    assert [(row["role"], row["content"]) for row in child_rows(parent)] == [
        (row["role"], row["content"]) for row in expected]
    assert module("aux_session").delegate_child_store_counts()["fallbacks"] > before
    assert any("counted identities" in record.message for record in caplog.records)


def test_restart_counts_compression_predecessors(host):
    load, clone = host
    ctx, _ = load()
    parent = clone(ctx)
    parent.on_session_start("parent", platform="cli", context_length=200_000)
    child = Agent("child", **DELEGATE)
    original = child_view(2)
    child.turn(ctx, original, parent)
    child.rotate(parent, "continued")
    ctx2, _ = load()  # fresh module and engines: no in-memory boundary or handoff map
    resumed = clone(ctx2)
    resumed.on_session_start("parent", platform="cli", context_length=200_000)
    view = child_view() + [{"role": "assistant", "content": "after restart"}]
    for _ in range(2):
        child.turn(ctx2, view, resumed)
    view.append({"role": "user", "content": "next turn"})
    child.turn(ctx2, view, resumed)
    assert [row["content"] for row in child_rows(resumed)] == [row["content"] for row in original]
    assert [row["content"] for row in child_rows(resumed, "continued")] == ["after restart", "next turn"]
    assert ("continued", "child", "compression") in lineage_rows(resumed)


def test_store_during_compression_does_not_publish_stale_boundary(host, monkeypatch):
    load, clone = host
    ctx, _ = load()
    parent, other = clone(ctx), clone(ctx)
    for engine in (parent, other):
        engine.on_session_start("parent", platform="cli", context_length=200_000)
    child = Agent("child", **DELEGATE)
    child.start(parent)
    original = child_view(3)
    concurrent = {"role": "user", "content": "concurrently stored turn"}
    compacted = [{"role": "user", "content": "compacted head"}, original[-1]]

    def compress_with_store(*args):
        assert other._store_delegate_child_turn("child", [*original, concurrent], platform="subagent") == 1
        return compacted

    monkeypatch.setattr(parent, "_compress_bypassed", compress_with_store)
    result = child.compress(parent, original)
    later = {"role": "assistant", "content": "later turn"}
    child.turn(ctx, [*result, concurrent, later], parent)
    child.turn(ctx, [*result, concurrent, later], parent)
    assert [row["content"] for row in child_rows(parent)] == [
        row["content"] for row in [*original, concurrent, later]]


def test_failed_compress_pre_store_does_not_publish_boundary(host, monkeypatch, caplog):
    load, clone = host
    ctx, module = load()
    engine = clone(ctx)
    engine.on_session_start("parent", platform="cli", context_length=200_000)
    child = Agent("child", **DELEGATE)
    original = [{"role": "user", "content": f"row {i}"} for i in range(1, 11)]
    child.turn(ctx, original[:5], engine)
    append = engine._append_off_current_session_end_suffix
    attempts = []

    def fail_once(*args, **kwargs):
        attempts.append(args)
        if len(attempts) == 1:
            raise sqlite3.OperationalError("synthetic append failure")
        return append(*args, **kwargs)

    compacted = [{"role": "user", "content": "compressed head"}, *original[-2:]]
    monkeypatch.setattr(engine, "_append_off_current_session_end_suffix", fail_once)
    monkeypatch.setattr(engine, "_compress_bypassed", lambda *args: compacted)
    with caplog.at_level("WARNING"):
        result = child.compress(engine, original)
    aux = module("aux_session")
    assert engine._delegate_child_key("child") not in aux._DELEGATE_RESUME_BOUNDARIES
    assert any("compression boundary not recorded: input not stored" in record.message for record in caplog.records)
    later = {"role": "user", "content": "row 11"}
    for _ in range(2):
        child.turn(ctx, [*result, later], engine)
    contents = [row["content"] for row in child_rows(engine)]
    assert all(contents.count(f"row {i}") == 1 for i in (9, 10, 11))


@pytest.mark.parametrize("shape", ["short", "tool-group"])
def test_stale_replay_does_not_rewind_compression_boundary(host, monkeypatch, shape):
    load, clone = host
    ctx, _ = load()
    engine = clone(ctx)
    engine.on_session_start("parent", platform="cli", context_length=200_000)
    child = Agent("child", **DELEGATE)
    original = child_view(2)
    compacted = [{"role": "user", "content": "compressed head"}, original[-1]]
    monkeypatch.setattr(engine, "_compress_bypassed", lambda *args: compacted)
    compacted = child.compress(engine, original)
    if shape == "short":
        additions = [{"role": "assistant", "content": "new answer"}]
        stale = compacted
    else:
        additions = [
            {"role": "user", "content": "next request"},
            {"role": "assistant", "content": "checking", "tool_calls": [
                {"id": "c6", "type": "function", "function": {"name": "terminal", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "c6", "content": "new tool result"},
            {"role": "assistant", "content": "new final answer"},
        ]
        stale = [*compacted, *additions[:2]]
    newest = {"role": "user", "content": "newest request"}
    child.turn(ctx, [*compacted, *additions], engine)
    child.turn(ctx, stale, engine)
    child.turn(ctx, [*compacted, *additions, newest], engine)
    child.turn(ctx, [*compacted, *additions, newest], engine)
    contents = [row["content"] for row in child_rows(engine)]
    assert all(contents.count(row["content"]) == 1 for row in [*additions, newest])


def test_compression_handoff_retries_missing_predecessor_link(host, monkeypatch):
    load, clone = host
    ctx, _ = load()
    engine = clone(ctx)
    engine.on_session_start("parent", platform="cli", context_length=200_000)
    child = Agent("child", **DELEGATE)
    child.start(engine)
    ctx.hooks["subagent_start"][0](child_session_id="child", parent_session_id="parent")
    record = engine._store.record_session_lineage
    attempts = []

    def fail_once(session_id, *args, **kwargs):
        attempts.append(session_id)
        if len(attempts) == 1:
            raise sqlite3.OperationalError("synthetic lineage contention")
        return record(session_id, *args, **kwargs)

    original = child_view(2)
    compacted = [{"role": "user", "content": "compressed head"}, *original[-2:]]
    monkeypatch.setattr(engine._store, "record_session_lineage", fail_once)
    monkeypatch.setattr(engine, "_compress_bypassed", lambda *args: compacted)
    result = child.compress(engine, original)
    assert lineage_rows(engine) == []
    child.rotate(engine, "continued")
    later = {"role": "assistant", "content": "continued answer"}
    for _ in range(2):
        child.turn(ctx, [*result, later], engine)
    assert [row["content"] for row in child_rows(engine)] == [row["content"] for row in original]
    assert [row["content"] for row in child_rows(engine, "continued")] == [later["content"]]
    assert sorted(lineage_rows(engine)) == [
        ("child", "parent", "subagent_start"), ("continued", "child", "compression")]
    assert attempts == ["child", "child", "continued"]


@pytest.mark.parametrize("counted", [False, True], ids=["anchor", "counter"])
def test_restart_matches_externalized_predecessor_rows(host, monkeypatch, counted):
    monkeypatch.setenv("LCM_LARGE_OUTPUT_EXTERNALIZATION_ENABLED", "true")
    monkeypatch.setenv("LCM_LARGE_OUTPUT_EXTERNALIZATION_THRESHOLD_CHARS", "256")
    load, clone = host
    ctx, _ = load()
    engine = clone(ctx)
    engine.on_session_start("parent", platform="cli", context_length=200_000)
    child = Agent("child", **DELEGATE)
    original = child_view()
    original[-1] = {**original[-1], "content": "large tool result line\n" * 1000}
    child.turn(ctx, original, engine)
    stored_before = child_rows(engine)
    assert stored_before[-1]["content"] != original[-1]["content"]  # the predecessor owns the payload ref
    child.rotate(engine, "continued")
    ctx2, _ = load()
    resumed = clone(ctx2)
    resumed.on_session_start("parent", platform="cli", context_length=200_000)
    if counted:
        monkeypatch.setattr(resumed, "_delegate_child_anchor_end", lambda *args: None)
    later = {"role": "assistant", "content": "after externalized tail"}
    for _ in range(2):
        child.turn(ctx2, [*original, later], resumed)
    assert child_rows(resumed) == stored_before
    assert [row["content"] for row in child_rows(resumed, "continued")] == [later["content"]]


def test_delegate_fallback_reads_real_tail_and_uncapped_rows(host, monkeypatch):
    load, clone = host
    ctx, _ = load()
    engine = clone(ctx)
    engine.on_session_start("parent", platform="cli", context_length=200_000)
    child = Agent("child", **DELEGATE)
    original = [{"role": "user", "content": f"stored row {i}"} for i in range(8)]
    child.turn(ctx, original, engine)
    read = engine._store.get_session_messages
    tail = engine._store.get_session_tail
    limits, tail_reads = [], []

    def capped_read(session_id, limit=2):
        limits.append(limit)
        return read(session_id, limit=limit)

    def track_tail(session_id, limit=1000):
        tail_reads.append((session_id, limit))
        return tail(session_id, limit=limit)

    monkeypatch.setattr(engine._store, "get_session_messages", capped_read)
    monkeypatch.setattr(engine._store, "get_session_tail", track_tail)
    first = {"role": "user", "content": "after true tail"}
    child.turn(ctx, [*original[-2:], first], engine)
    assert ("child", 3) in tail_reads
    assert len(read("child", limit=-1)) == len(original) + 1
    monkeypatch.setattr(engine, "_delegate_child_anchor_end", lambda *args: None)
    second = {"role": "user", "content": "after counted fallback"}
    for _ in range(2):
        child.turn(ctx, [original[4], first, second], engine)
    assert -1 in limits
    assert [row["content"] for row in read("child", limit=-1)] == [
        row["content"] for row in [*original, first, second]]


@pytest.mark.parametrize("parent_kind", ["ignored", "delegate"])
def test_stateless_rotated_parent_requires_delegate_identity(host, monkeypatch, parent_kind):
    monkeypatch.setenv("LCM_IGNORE_SESSION_PATTERNS", "old-parent")
    load, clone = host
    ctx, _ = load()
    engine = clone(ctx)
    if parent_kind == "ignored":
        engine.on_session_start("old-parent", platform="cli", context_length=200_000)
        old_id = "old-parent"
    else:
        engine.on_session_start("root", platform="cli", context_length=200_000)
        Agent("old-child", parent="root", **DELEGATE).turn(ctx, child_view(), engine)
        # Rehydrate the child's bypass lineage on a separate engine before rotation.
        engine = clone(ctx)
        engine.on_session_start("old-child", platform="subagent", context_length=200_000)
        engine._mark_lcm_bypass_lineage_session("old-child", platform="subagent")
        old_id = "old-child"
    engine.on_session_start("parent", platform="subagent", boundary_reason="compression", old_session_id=old_id)
    assert engine._session_id == "parent" and engine._session_stateless and not engine._session_ignored
    links_before = lineage_rows(engine)
    child = Agent("child", **DELEGATE)
    child.turn(ctx, child_view(), engine)
    if parent_kind == "ignored":
        assert child_rows(engine) == [] and lineage_rows(engine) == []
    else:
        assert [row["content"] for row in child_rows(engine)] == [row["content"] for row in child_view()]
        assert lineage_rows(engine) == [*links_before, ("child", "parent", "parent_frame")]
