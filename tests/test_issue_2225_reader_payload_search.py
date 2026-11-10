"""#2225: bounded saved-output search through the read-only MCP endpoint."""

import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_lcm import mcp_reader, tools
from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryDAG
from hermes_lcm.db_bootstrap import join_background_stem_backfills
from hermes_lcm.externalize import extract_externalized_ref, get_large_output_storage_dir
from hermes_lcm.store import MessageStore

TOKEN = "buried2225fact"


class Archive:
    def __init__(self, home):
        home.mkdir()
        self.home = home
        self.config = LCMConfig(database_path=str(home / "lcm.db"), embeddings_enabled=False,
                                large_output_externalization_threshold_chars=12000)
        self.store = MessageStore(self.config.database_path, ingest_protection_config=self.config,
                                  hermes_home=str(home))
        self.dag = SummaryDAG(self.config.database_path)
        self.reader = mcp_reader.StoreReader(self.config.database_path, home, config=self.config)
        self.storage = get_large_output_storage_dir(self.config, hermes_home=str(home), create=True)

    def save(self, session="saved", content=None, *, name=None, created_at=None):
        content = content if content is not None else "padding " * 1700 + "\n" + TOKEN
        row_id = self.store.append(session, {"role": "tool", "content": content,
                                             "tool_call_id": f"call-{session}"})
        row = self.store.get(row_id)
        ref = extract_externalized_ref(row["content"])
        assert ref and TOKEN not in row["content"]
        path = self.storage / ref
        if created_at is not None:
            payload = json.loads(path.read_text())
            payload["created_at"] = created_at
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
        if name:
            path.rename(self.storage / name)
            ref = name
        return ref, content

    def call(self, name="lcm_grep", **args):
        raw, error = self.reader.call(name, args)
        result = json.loads(raw)
        assert not error, result
        return result

    def grep(self, **args):
        return self.call(query=TOKEN, **args)

    def close(self):
        self.store.close()
        self.dag.close()
        join_background_stem_backfills(timeout=10)


@pytest.fixture
def archive(tmp_path):
    archive = Archive(tmp_path / "store")
    yield archive
    archive.close()


def payload_hits(result):
    return [hit for hit in result["results"] if hit["type"] == "externalized"]


def test_default_reader_finds_ingested_payload_and_expands_byte_equal(archive):
    ref, content = archive.save()
    result = archive.grep()
    hit, = payload_hits(result)
    assert hit["session_id"] == "saved" and hit["from_current_session"] is False
    assert hit["ref"] == ref and TOKEN in hit["snippet"]
    assert hit["line"] == 2 and hit["byte_position"] == content.encode().index(TOKEN.encode())
    assert "partial" not in result
    expanded = archive.call("lcm_expand", externalized_ref=ref, session_id=hit["session_id"], max_tokens=20000)
    assert expanded["content"].encode() == content.encode()
    assert expanded["has_more"] is False


@pytest.mark.parametrize("scope", [None, "externalized", "both"])
def test_session_scope_excludes_descendants_from_history_and_payloads(archive, scope):
    archive.save("child")
    archive.store.record_session_lineage("child", "parent", source="subagent")
    archive.store.append("child", {"role": "user", "content": TOKEN})
    args = {} if scope is None else {"content_scope": scope}
    result = archive.grep(session_scope="session", session_id="parent", **args)
    assert result["results"] == []
    assert not archive.grep(session_scope="session", session_id="other", **args)["results"]


@pytest.mark.parametrize("scope", ["externalized", "both"])
def test_explicit_all_scope(archive, scope):
    ref, _ = archive.save()
    assert [h["ref"] for h in payload_hits(archive.grep(content_scope=scope, session_scope="all"))] == [ref]


def test_explicit_history_and_current_remain_history_only(archive):
    archive.save()
    assert archive.grep(content_scope="history")["results"] == []
    assert archive.grep(session_scope="current")["results"] == []
    raw, error = archive.reader.call("lcm_grep", {"query": TOKEN, "session_scope": "current",
                                                "content_scope": "both"})
    assert error and json.loads(raw)["error"] == "Externalized payload search requires an active session"


def test_default_history_full_skips_payloads_under_limit_fills_remaining(archive, monkeypatch):
    archive.save()
    archive.store.append("history", {"role": "user", "content": TOKEN})
    with monkeypatch.context() as patch:
        def forbidden(*args, **kwargs):
            pytest.fail("history filled the limit; payload pass must not run")
        patch.setattr(tools, "get_large_output_storage_dir", forbidden)
        full = archive.grep(limit=1)
    assert full["results"][0]["type"] == "message" and "partial" not in full
    assert full["externalized_skipped"] == "history_filled_limit"
    archive.save("second")
    result = archive.grep(limit=2)
    assert len(result["results"]) == 2
    assert len(payload_hits(result)) == 1
    assert any(h["type"] == "message" for h in result["results"])


def test_file_cap_newest_first_and_scope_total(archive, monkeypatch):
    archive.save("saved", name="20261109_120000_old.json", created_at=1)
    newest, _ = archive.save("saved", name="20261110_120000_new.json", created_at=3)
    archive.save("other", name="20261108_120000_other.json", created_at=2)
    monkeypatch.setattr(tools, "READER_PAYLOAD_FILE_CAP", 1, raising=False)
    result = archive.grep(session_scope="session", session_id="saved")
    assert [h["ref"] for h in payload_hits(result)] == [newest]
    assert result["externalized_scan"]["file_limit"] == 1
    assert result["partial"] == {"payloads_searched": 1, "payloads_total": 3,
                                  "reasons": ["file_cap"], "prefix_bytes": 512000}


def test_byte_cap_never_overreads_and_reports_partial(archive, monkeypatch):
    archive.save(created_at=1)
    archive.save("second", created_at=2)
    monkeypatch.setattr(tools, "READER_PAYLOAD_BYTE_CAP", 14000, raising=False)
    result = archive.grep()
    assert not payload_hits(result)  # Header and tail consume the small budget first.
    assert result["partial"] == {"payloads_searched": 1, "payloads_total": 2,
                                  "reasons": ["byte_cap"], "prefix_bytes": 512000}


def test_prefix_cut_is_disclosed_even_without_hit(archive):
    archive.save(content="x" * 512010 + TOKEN)
    result = archive.grep()
    assert not result["results"]
    assert result["partial"] == {"payloads_searched": 1, "payloads_total": 1,
                                  "reasons": ["prefix"], "prefix_bytes": 512000}


def test_discovery_cap_is_disclosed(archive, monkeypatch):
    archive.save()
    archive.save("second")
    monkeypatch.setattr(tools, "_LCM_GREP_EXTERNALIZED_DISCOVERY_CAP", 1)
    result = archive.grep()
    assert result["partial"] == {"payloads_searched": 1, "payloads_total": ">4096",
                                  "reasons": ["discovery_cap"], "prefix_bytes": 512000}


def test_response_cap_is_disclosed(archive, monkeypatch):
    archive.save()
    monkeypatch.setattr(tools, "_LCM_GREP_RESPONSE_CHAR_CAP", 1)
    result = archive.grep()
    assert result["results"] == []
    assert result["partial"]["reasons"] == ["response_cap"]


def test_second_store_is_never_read(archive, tmp_path, monkeypatch):
    other = Archive(tmp_path / "other")
    try:
        other.config.large_output_externalization_path = str(archive.storage)
        other.storage = archive.storage
        foreign_ref, _ = other.save("foreign")
        real_open = tools.os.open

        def guarded(path, *args, **kwargs):
            assert not str(path).startswith(str(other.home))
            return real_open(path, *args, **kwargs)

        monkeypatch.setattr(tools.os, "open", guarded)
        result = archive.grep()
        assert not result["results"] and "partial" not in result
        assert foreign_ref not in [hit["ref"] for hit in payload_hits(result)]
        archive.save()
        assert len(payload_hits(archive.grep())) == 1
    finally:
        other.close()


@pytest.mark.parametrize("ref", ["../outside.json", "sub/file.json", "sub\\file.json", "wrong.txt", "linked.json"])
def test_unsafe_explicit_ref_is_rejected(archive, ref):
    target, _ = archive.save()
    (archive.storage / "linked.json").symlink_to(archive.storage / target)
    raw, error = archive.reader.call("lcm_grep", {"query": TOKEN, "content_scope": "both",
                                                "externalized_refs": [ref]})
    assert error and "error" in json.loads(raw)
    if ref == "linked.json":
        assert "symlink" in json.loads(raw)["error"]
        assert [h["ref"] for h in payload_hits(archive.grep())] == [target]


def test_explicit_refs_use_scope_and_reader_caps(archive, monkeypatch):
    old, _ = archive.save(name="20261109_120000_old.json", created_at=1)
    new, _ = archive.save("second", name="20261110_120000_new.json", created_at=2)
    monkeypatch.setattr(tools, "READER_PAYLOAD_FILE_CAP", 1, raising=False)
    result = archive.grep(content_scope="externalized", externalized_refs=[old, new])
    assert [h["ref"] for h in payload_hits(result)] == [new]
    assert result["partial"]["reasons"] == ["file_cap"]
    raw, error = archive.reader.call("lcm_grep", {"query": TOKEN, "content_scope": "both",
                                                "externalized_refs": [new],
                                                "session_scope": "session", "session_id": "saved"})
    assert error and "owned" in json.loads(raw)["error"]


def test_reader_description_mentions_payloads_and_partial():
    definition = next(d for d in mcp_reader.tool_definitions(set()) if d["name"] == "lcm_grep")
    assert "Large tool outputs are searched too" in definition["description"]
    assert "partial" in definition["description"]


@pytest.mark.parametrize("scope,args", [("all", {}), ("session", {"session_id": "saved"})])
def test_nonreader_cross_session_refusal_is_byte_identical(archive, scope, args):
    engine = SimpleNamespace(_config=archive.config, _hermes_home=str(archive.home),
                             _store=archive.store, _dag=archive.dag, current_session_id="saved")
    assert tools.lcm_grep({"query": TOKEN, "content_scope": "both", "session_scope": scope, **args},
                          engine=engine) == '{"error": "Externalized payload search supports session_scope=current only"}'


def test_nonreader_default_is_byte_identical_history_only(archive):
    archive.save()
    engine = SimpleNamespace(_config=archive.config, _hermes_home=str(archive.home),
                             _store=archive.store, _dag=archive.dag, current_session_id="saved")
    assert tools.lcm_grep({"query": TOKEN}, engine=engine) == json.dumps({
        "query": TOKEN, "sort": "recency", "session_scope": "current", "content_scope": "history",
        "source": None, "conversation_id": None, "limit": 10, "total_results": 0, "results": [],
    })


def test_multiword_retry_preserves_partial_and_scans_payloads_once(archive, monkeypatch):
    archive.save()
    archive.save("second")
    monkeypatch.setattr(tools, "READER_PAYLOAD_FILE_CAP", 1, raising=False)
    real_read = tools._reader_payload_search
    reads = []

    def record(*args, **kwargs):
        reads.append(args[0])
        return real_read(*args, **kwargs)

    monkeypatch.setattr(tools, "_reader_payload_search", record)
    result = archive.call(query="missing2225 phrase2225")
    assert result["retried"] == "any_term" and not result["results"]
    assert result["partial"] == {"payloads_searched": 1, "payloads_total": 2,
                                  "reasons": ["file_cap"], "prefix_bytes": 512000}
    assert len(reads) == 1


def test_exact_prefix_boundary_is_complete(archive):
    archive.save(content="x" * (512000 - len(TOKEN)) + TOKEN)
    result = archive.grep()
    assert len(payload_hits(result)) == 1
    assert "partial" not in result


def test_small_byte_budget_must_also_fit_metadata(archive, monkeypatch):
    _, content = archive.save(content="padding " * 1700 + TOKEN)
    monkeypatch.setattr(tools, "READER_PAYLOAD_BYTE_CAP", len(content.encode()), raising=False)
    result = archive.grep()
    assert not payload_hits(result)
    assert result["partial"]["reasons"] == ["byte_cap"]


@pytest.mark.parametrize("field,value", [("session_id", ""), ("session_id", "outside"), ("kind", None)])
def test_discovery_rejects_invalid_metadata_and_enforces_scope(archive, field, value):
    ref, _ = archive.save()
    path = archive.storage / ref
    document = json.loads(path.read_text())
    if field == "kind":
        # Invalid document structure must not turn into a searchable prefix.
        path.write_text(path.read_text()[:-2] + "INVALID")
    else:
        document[field] = value
        path.write_text(json.dumps(document, indent=2))
    assert archive.grep(session_scope="session", session_id="saved")["results"] == []


def test_metadata_open_rejects_payload_replaced_by_symlink(archive, monkeypatch):
    ref, _ = archive.save()
    path = archive.storage / ref
    outside = archive.home / "outside.json"
    outside.write_bytes(path.read_bytes())
    real_open = tools.os.open
    replaced = []

    def replace_on_open(target, *args, **kwargs):
        if str(target) in {str(path), path.name} and not replaced:
            path.unlink()
            path.symlink_to(outside)
            replaced.append(True)
        return real_open(target, *args, **kwargs)

    monkeypatch.setattr(tools.os, "open", replace_on_open)
    assert archive.grep()["results"] == []
    assert replaced


@pytest.mark.parametrize("args", [{"source": "claude-code"}, {"role": "tool"},
                                   {"conversation_id": "thread"}, {"time_from": 1}])
def test_reader_payloads_do_not_bypass_history_filters(archive, args):
    archive.save()
    result = archive.grep(**args)
    assert result["results"] == []
    assert result["externalized_results_omitted"] is True
    assert "partial" not in result


def test_prefix_cut_does_not_trust_claimed_content_size(archive):
    ref, _ = archive.save(content="x" * 512010 + TOKEN)
    path = archive.storage / ref
    payload = json.loads(path.read_text())
    payload["content_bytes"] = 1
    path.write_text(json.dumps(payload, indent=2))
    result = archive.grep()
    assert result["results"] == []
    assert result["partial"]["reasons"] == ["prefix"]


@pytest.mark.parametrize("name,mode", [("lcm_grep", "hybrid"), ("lcm_grep", "semantic"),
                                      ("lcm_recall", None)])
def test_internal_searches_never_open_payloads_and_keep_history_parity(archive, monkeypatch, name, mode):
    archive.save()
    archive.store.append("saved", {"role": "user", "content": TOKEN})
    args = {"query": TOKEN, "session_scope": "all"}
    if mode:
        args["mode"] = mode
    now = time.time()
    monkeypatch.setattr(time, "time", lambda: now)
    with archive.reader._open_call() as (engine, _):
        engine._reader_any_session = False
        expected = getattr(tools, name)(args, engine=engine)
        engine._reader_any_session = True
        def forbidden(*args, **kwargs):
            pytest.fail("internal search attempted payload discovery")
        monkeypatch.setattr(tools, "get_large_output_storage_dir", forbidden)
        started = time.monotonic()
        actual = getattr(tools, name)(args, engine=engine)
        assert time.monotonic() - started < 1
        assert actual == expected


def test_reader_history_scope_keeps_parent_only_parity(archive):
    archive.save("child")
    archive.store.record_session_lineage("child", "parent", source="subagent")
    archive.store.append("child", {"role": "user", "content": TOKEN})
    archive.store.append("parent", {"role": "user", "content": TOKEN})
    args = {"query": TOKEN, "session_scope": "session", "session_id": "parent",
            "content_scope": "history"}
    with archive.reader._open_call() as (engine, _):
        engine._reader_any_session = False
        expected = tools.lcm_grep(args, engine=engine)
        engine._reader_any_session = True
        assert tools.lcm_grep(args, engine=engine) == expected


@pytest.mark.parametrize("location", ["outside", "profile"])
def test_storage_containment_skips_before_open(archive, tmp_path, monkeypatch, location):
    archive.save()
    target = tmp_path / "outside" if location == "outside" else archive.home / "profiles" / "named"
    target.mkdir(parents=True)
    link = archive.home / "linked"
    link.symlink_to(target, target_is_directory=True)
    archive.reader.config.large_output_externalization_path = str(link)  # the reader keeps its own copy of the config
    real_scandir = tools.os.scandir
    def guarded(path="."):
        # Reader discovery lists with os.scandir; the containment check may list the profiles folder, so only
        # listing the payload dir is discovery.
        if not isinstance(path, int) and Path(path).resolve() == target.resolve():  # an int is a dir fd (rmtree)
            pytest.fail("containment failure must skip discovery")
        return real_scandir(path)
    monkeypatch.setattr(tools.os, "scandir", guarded)
    result = archive.grep()
    assert result["results"] == []
    assert result["partial"]["reasons"] == ["containment"]


def test_discovery_is_lazy_and_counts_all_reads(archive, monkeypatch):
    for i in range(8):
        archive.save(name=f"20261110_12000{i}_payload.json")
    monkeypatch.setattr(tools, "READER_PAYLOAD_FILE_CAP", 1)
    monkeypatch.setattr(tools, "READER_PAYLOAD_BYTE_CAP", 14000)
    reads, opened = [], []
    original = tools.os.fdopen
    class CountingFile:
        def __init__(self, handle):
            self.handle = handle
        def __enter__(self):
            return self
        def __exit__(self, *args):
            self.handle.close()
        def __getattr__(self, name):
            return getattr(self.handle, name)
        def read(self, size=-1):
            data = self.handle.read(size)
            reads.append(len(data))
            return data
    def counted(*args, **kwargs):
        opened.append(args[0])
        return CountingFile(original(*args, **kwargs))
    monkeypatch.setattr(tools.os, "fdopen", counted)
    result = archive.grep()
    assert len(opened) <= 1
    assert sum(reads) <= 14000
    assert "byte_cap" in result["partial"]["reasons"]


@pytest.mark.parametrize("with_target", [True, False])
def test_out_of_scope_payloads_do_not_spend_file_cap(archive, with_target, monkeypatch):
    target, _ = archive.save("target", name="20261101_120000_target.json")
    document = json.loads((archive.storage / target).read_text())
    # One other session has rows here; the other belongs to a foreign store.
    archive.store.append("other", {"role": "user", "content": "other session"})
    for i in range(200):
        document["session_id"] = "other" if i % 2 else "foreign"
        (archive.storage / f"20261110_120000_reject{i:03}.json").write_text(json.dumps(document, indent=2))
    if not with_target:
        document["session_id"] = "foreign"
        (archive.storage / target).write_text(json.dumps(document, indent=2))
    searched = []
    original = tools._reader_payload_search
    def record(*args):
        result = original(*args)
        searched.append(result)
        return result
    monkeypatch.setattr(tools, "_reader_payload_search", record)
    result = archive.grep(session_scope="session", session_id="target")
    assert [hit["ref"] for hit in payload_hits(result)] == ([target] if with_target else [])
    assert "partial" not in result
    assert result["externalized_scan"]["candidate_files"] == 201
    assert result["externalized_scan"]["scanned_files"] == int(with_target)
    assert result["externalized_scan"]["rejected_session_mismatch"] == 201 - int(with_target)
    assert result["externalized_scan"]["rejected_invalid_or_unreadable"] == 0
    assert sum(item["status"] != "out_of_scope" for item in searched) == int(with_target)
    # Rejected headers still consume bytes, and the discovered total stays intact.
    monkeypatch.setattr(tools, "READER_PAYLOAD_BYTE_CAP", 8192)
    result = archive.grep(session_scope="session", session_id="target")
    assert result["partial"] == {"payloads_searched": 0, "payloads_total": 201,
                                  "reasons": ["byte_cap"], "prefix_bytes": 512000}


def test_missing_payload_directory_skips_containment_quietly(archive, monkeypatch):
    (archive.home / "profiles").mkdir()
    missing = archive.storage / "not-created"
    monkeypatch.setattr(tools, "get_large_output_storage_dir", lambda *args, **kwargs: missing)
    result = archive.grep()
    assert result["results"] == []
    assert "partial" not in result
    assert not missing.exists()


@pytest.mark.parametrize("unsupported", ["flags", "dir_fd"])
def test_reader_payload_open_portable_fallback(archive, monkeypatch, unsupported):
    ref, _ = archive.save()
    original = tools.os.open
    def portable_open(path, flags, *args, **kwargs):
        if unsupported == "dir_fd":
            assert kwargs.get("dir_fd") is None
        return original(path, flags, *args, **kwargs)
    # tools.os is the process-wide os module: undo the patches before any teardown (rmtree uses them).
    with monkeypatch.context() as patch:
        if unsupported == "flags":
            patch.delattr(tools.os, "O_DIRECTORY")
            patch.delattr(tools.os, "O_NOFOLLOW")
        else:
            patch.setattr(tools.os, "supports_dir_fd", set())
        patch.setattr(tools.os, "open", portable_open)
        result = archive.grep()
    assert [hit["ref"] for hit in payload_hits(result)] == [ref]
    assert "partial" not in result


def test_explicit_refs_keep_earlier_matches_when_the_budget_runs_out(archive, monkeypatch):
    # Bot review of #2528: a budget cut inside an explicit ref is byte_cap, not an unreadable ref.
    first, _ = archive.save(session="saved", name="20261110_120002_first.json")
    second, _ = archive.save(session="saved", name="20261110_120001_second.json",
                             content="padding " * 30000 + "\n" + TOKEN)
    spent = []
    original = tools._reader_payload_search
    def measured(*args, **kwargs):
        result = original(*args, **kwargs)
        spent.append(result["bytes_read"])
        return result
    monkeypatch.setattr(tools, "_reader_payload_search", measured)
    archive.grep(content_scope="externalized", externalized_refs=[first])
    monkeypatch.setattr(tools, "READER_PAYLOAD_BYTE_CAP", spent[0] + 64)  # the first fits; the second is cut
    result = archive.grep(content_scope="externalized", externalized_refs=[first, second])
    assert "error" not in result
    assert [hit["ref"] for hit in payload_hits(result)] == [first]
    assert result["partial"]["reasons"] == ["byte_cap"]


def test_any_term_retry_with_explicit_refs_is_not_an_error(archive):
    ref, _ = archive.save()
    result = archive.call(query="absentword otherabsent", content_scope="both", externalized_refs=[ref])
    assert "error" not in result and result["retried"] == "any_term"
    assert result["externalized_refs"] == [ref]


def test_reader_schema_advertises_its_own_content_scope(archive):
    from hermes_lcm import schemas
    grep = next(tool for tool in mcp_reader.tool_definitions(set()) if tool["name"] == "lcm_grep")
    scope = grep["inputSchema"]["properties"]["content_scope"]
    assert "default" not in scope and "session_scope 'all' or 'session'" in scope["description"]
    assert "active session" not in grep["inputSchema"]["properties"]["externalized_refs"]["description"]
    # Hermes' own schema object is unchanged.
    assert schemas.LCM_GREP["parameters"]["properties"]["content_scope"]["default"] == "history"
    assert "session_scope='current' only" in schemas.LCM_GREP["parameters"]["properties"]["content_scope"]["description"]


def test_a_confirmed_out_of_scope_ref_still_errors_at_the_budget_boundary(archive, monkeypatch):
    # Codex review of the bot-round delta: only a check the budget cut short becomes byte_cap. A ref whose header
    # was read in full and named another session stays an error, even when the budget ends right there.
    mine, _ = archive.save(session="mine", name="20261110_120002_mine.json")
    theirs, _ = archive.save(session="theirs", name="20261110_120001_theirs.json")
    spent = []
    original = tools._reader_payload_search
    def measured(*args, **kwargs):
        result = original(*args, **kwargs)
        spent.append((result["status"], result["bytes_read"]))
        return result
    monkeypatch.setattr(tools, "_reader_payload_search", measured)
    args = dict(content_scope="externalized", session_scope="session", session_id="mine")
    for ref in (mine, theirs):  # measured alone; the second is refused, as it should be
        archive.reader.call("lcm_grep", dict(query=TOKEN, externalized_refs=[ref], **args))
    (_, mine_bytes), (status, theirs_bytes) = spent[0], spent[1]
    assert status == "out_of_scope"
    monkeypatch.setattr(tools, "READER_PAYLOAD_BYTE_CAP", mine_bytes + theirs_bytes)  # ends exactly after its header
    raw, error = archive.reader.call("lcm_grep", dict(query=TOKEN, externalized_refs=[mine, theirs], **args))
    assert "not owned by the requested scope" in json.loads(raw).get("error", "")


def test_discovery_cap_keeps_the_newest_names_whatever_the_listing_order(archive, monkeypatch):
    # Bot review of #2528: the cap must keep the newest payloads, not the first ones the file system lists.
    refs = [archive.save(name=f"20261110_12000{i}_p.json")[0] for i in range(5)]
    monkeypatch.setattr(tools, "_LCM_GREP_EXTERNALIZED_DISCOVERY_CAP", 2)
    real_scandir = tools.os.scandir
    class OldestFirst:
        def __init__(self, path):
            with real_scandir(path) as entries:
                self.entries = sorted(entries, key=lambda entry: entry.name)
        def __enter__(self):
            return iter(self.entries)
        def __exit__(self, *exc):
            return False
    monkeypatch.setattr(tools.os, "scandir", OldestFirst)
    result = archive.grep()
    assert sorted(hit["ref"] for hit in payload_hits(result)) == sorted(refs[-2:])
    assert "discovery_cap" in result["partial"]["reasons"]


def test_a_payload_the_reader_cannot_parse_is_disclosed(archive):
    # Bot review of #2528: a valid-looking payload the reader cannot reconstruct is never a silent skip.
    good, _ = archive.save(name="20261110_120001_good.json")
    bad, _ = archive.save(name="20261110_120002_bad.json")
    path = archive.storage / bad
    path.write_bytes(path.read_bytes()[:-200])  # the closing quote and tail metadata are gone
    result = archive.grep()
    assert [hit["ref"] for hit in payload_hits(result)] == [good]
    assert "unreadable" in result["partial"]["reasons"]


@pytest.mark.parametrize("shape", ["invalid_utf8", "complete_malformed"])
def test_a_confirmed_unreadable_ref_still_errors_at_the_budget_boundary(archive, monkeypatch, shape):
    # Codex re-review of #2528: bytes already read that are malformed are a confirmed failure, and a read that
    # reaches the file's end is not a budget cut, so neither becomes a silent byte_cap.
    mine, _ = archive.save(session="mine", name="20261110_120002_mine.json")
    bad = "20261110_120001_bad.json"
    body = b'{"\xff\xfe' + b"x" * 5000 if shape == "invalid_utf8" else b'{"session_id": "mine", "kind": "x"}'
    (archive.storage / bad).write_bytes(body)
    spent = []
    original = tools._reader_payload_search
    def measured(*args, **kwargs):
        result = original(*args, **kwargs)
        spent.append(result["bytes_read"])
        return result
    monkeypatch.setattr(tools, "_reader_payload_search", measured)
    args = dict(content_scope="externalized", session_scope="session", session_id="mine")
    archive.reader.call("lcm_grep", dict(query=TOKEN, externalized_refs=[mine], **args))
    left = 32 if shape == "invalid_utf8" else len(body)  # a cut read, or a budget that ends at the file's end
    monkeypatch.setattr(tools, "READER_PAYLOAD_BYTE_CAP", spent[0] + left)
    raw, error = archive.reader.call("lcm_grep", dict(query=TOKEN, externalized_refs=[mine, bad], **args))
    assert "not readable" in json.loads(raw).get("error", "")


def test_a_payload_folder_that_cannot_be_listed_is_disclosed(archive, monkeypatch):
    # NeonDiff on #2528: a listing failure is not a complete search with no matches.
    archive.save()
    def refuse(path):
        raise PermissionError("denied")
    monkeypatch.setattr(tools.os, "scandir", refuse)
    result = archive.grep()
    assert payload_hits(result) == [] and "unreadable" in result["partial"]["reasons"]


def test_a_budget_cut_inside_a_multibyte_character_stays_byte_cap(archive, monkeypatch):
    # A cut that splits a character in a valid header is a budget cut, not a corrupt ref.
    mine, _ = archive.save(session="mine", name="20261110_120002_mine.json")
    wide, _ = archive.save(session="mine", name="20261110_120001_wide.json")
    path = archive.storage / wide
    path.write_text(json.dumps({"note": "\u00e9" * 3000, **json.loads(path.read_text())}, ensure_ascii=False, indent=2))
    spent = []
    original = tools._reader_payload_search
    def measured(*args, **kwargs):
        result = original(*args, **kwargs)
        spent.append(result["bytes_read"])
        return result
    monkeypatch.setattr(tools, "_reader_payload_search", measured)
    args = dict(content_scope="externalized", session_scope="session", session_id="mine")
    archive.reader.call("lcm_grep", dict(query=TOKEN, externalized_refs=[mine], **args))
    middle = path.read_bytes().index("\u00e9".encode()) + 2 * 50 + 1  # inside the 51st character
    monkeypatch.setattr(tools, "READER_PAYLOAD_BYTE_CAP", spent[0] + middle)
    raw, error = archive.reader.call("lcm_grep", dict(query=TOKEN, externalized_refs=[mine, wide], **args))
    result = json.loads(raw)
    assert "error" not in result and result["partial"]["reasons"] == ["byte_cap"]
    assert [hit["ref"] for hit in payload_hits(result)] == [mine]


def test_a_file_that_grows_after_it_is_opened_is_still_a_budget_cut(archive, monkeypatch):
    # Reviews of #2528: the size check uses the file's size at the cut, not at open, so a payload the in-place
    # writer is still growing is a budget cut, not a confirmed bad ref.
    mine, _ = archive.save(session="mine", name="20261110_120002_mine.json")
    wide, _ = archive.save(session="mine", name="20261110_120001_wide.json")
    spent = []
    original = tools._reader_payload_search
    def measured(*args, **kwargs):
        result = original(*args, **kwargs)
        spent.append(result["bytes_read"])
        return result
    monkeypatch.setattr(tools, "_reader_payload_search", measured)
    args = dict(content_scope="externalized", session_scope="session", session_id="mine")
    archive.reader.call("lcm_grep", dict(query=TOKEN, externalized_refs=[mine], **args))
    real_fstat, seen = tools.os.fstat, set()
    def stale_at_open(fd):  # the first look at each open file sees it empty, as if the writer had not written yet
        result = real_fstat(fd)
        if (result.st_dev, result.st_ino) in seen:
            return result
        seen.add((result.st_dev, result.st_ino))
        fields = list(result)
        fields[6] = 0
        return type(result)(fields)
    monkeypatch.setattr(tools.os, "fstat", stale_at_open)
    monkeypatch.setattr(tools, "READER_PAYLOAD_BYTE_CAP", spent[0] + 32)
    raw, error = archive.reader.call("lcm_grep", dict(query=TOKEN, externalized_refs=[mine, wide], **args))
    result = json.loads(raw)
    assert "error" not in result and result["partial"]["reasons"] == ["byte_cap"]


def test_a_storage_path_that_cannot_be_resolved_degrades_to_containment(archive, monkeypatch):
    # CodeRabbit on #2528: before Python 3.13 a symlink loop raises RuntimeError; history results must stand.
    ref, _ = archive.save()
    def loop(*args, **kwargs):
        raise RuntimeError("Symlink loop")
    monkeypatch.setattr(tools, "get_large_output_storage_dir", loop)
    result = archive.call(query="call-saved")
    assert "error" not in result and result["partial"]["reasons"] == ["containment"]
