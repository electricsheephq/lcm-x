"""Offline #995 arms, using two synthetic conversations and pinned-base goldens."""
import io
import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

from bench.fmp import common, datasets, delivery, mcp_reader, models, run, score, stores
from bench.fmp.tests.test_score_commits import COMMITS, arm

REPO = Path(__file__).resolve().parents[3]
BASE = "458d1b84108991cbddfbc0295e6bc257f2362c16"
VARIANTS = ("speaker-prefix", "plain", "name-date")


@pytest.fixture(scope="module")
def lme():
    module = common.load_lcm(REPO)
    from hermes_lcm import tokens
    tokens._encoder_ready, tokens._encoder = True, None
    return module


def conversations():
    return [datasets.normalize_locomo({"sample_id": f"synthetic-{i}", "conversation": {
        "speaker_a": "Ada", "speaker_b": "Bo",
        "session_1_date_time": "2025-01-02T00:30:00+02:00",
        "session_1": [{"speaker": "Ada", "text": f"teal key {i}"},
                      {"speaker": "Bo", "text": "Ada: keep this literal prefix"}],
        "session_2_date_time": "2025-01-03T12:00:00+00:00",
        "session_2": [{"speaker": "Ada", "text": "key in the drawer"}]},
        "qa": [{"category": 1, "question": "teal key?", "answer": "drawer"}]})[0] for i in range(2)]


def make_store(tmp_path, lme, monkeypatch, variant=None):
    monkeypatch.setattr(stores, "commit", lambda _: BASE)
    monkeypatch.setattr(delivery.time, "time", lambda: 1735689600.0)
    # Freeze SQLite's independent wall clock too, without changing schema or product code.
    connect = sqlite3.connect
    def frozen_connect(*args, **kwargs):
        connection = connect(*args, **kwargs)
        connection.create_function("strftime", 2, lambda fmt, date: "1735689600")
        return connection
    monkeypatch.setattr(sqlite3, "connect", frozen_connect)
    options = {} if variant is None else {"store_text": variant}
    return stores.Stores(tmp_path, "off", REPO, "synthetic-data", lme, "deterministic", **options)


def snapshot(store, q):
    store.build(q)
    directory = store.directory(q)
    speakers = json.loads((directory / "speakers.json").read_text())
    with store.engine(q) as (engine, _):
        raw = json.dumps({"hits": [{"store_id": i} for i in (2, 1, 3)]})
        from unittest.mock import patch
        with patch.object(delivery, "dispatch", return_value=raw):
            evidence, returned = delivery.single(q, engine, speakers)
        assert returned == raw
    return {"db_sha256": common.digest_file(directory / "lcm.db"),
            "speakers": (directory / "speakers.json").read_text(), "delivery": evidence}


def test_default_golden(tmp_path, lme, monkeypatch):
    args = cli(tmp_path)
    assert (args.store_text, args.hit_label) == ("speaker-prefix", "speaker")
    store = make_store(tmp_path / "stores", lme, monkeypatch)
    got = [snapshot(store, q) for q in conversations()]
    assert got == json.loads(Path(__file__).with_name("switches-golden.json").read_text())
    explicit = make_store(tmp_path / "explicit", lme, monkeypatch, "speaker-prefix")
    assert [snapshot(explicit, q) for q in conversations()] == got


@pytest.mark.parametrize("variant", VARIANTS)
def test_store_text_and_all_ingestion_consumers(tmp_path, lme, monkeypatch, variant):
    q = conversations()[0]
    store = make_store(tmp_path, lme, monkeypatch, variant)
    store.provider_name, store.model = "synthetic", "synthetic-embedding"
    embedded = []
    store.provider = SimpleNamespace(dim=2, embed_documents=lambda texts: embedded.extend(texts) or [[1., 0.] for _ in texts])
    assert store.build(q) == "built"
    expected = {"speaker-prefix": "Ada: teal key 0", "plain": "teal key 0",
                "name-date": "Ada (2025-01-01): teal key 0"}[variant]
    with store.engine(q) as (engine, vectors):
        assert engine._store.get_batch([1])[1]["content"] == expected
        assert expected in embedded
        summaries = engine._dag._conn.execute("SELECT summary FROM summary_nodes").fetchall()
        assert any(expected in row[0] for row in summaries)
        assert engine._store._conn.execute("SELECT count(*) FROM messages_fts WHERE messages_fts MATCH 'teal'").fetchone()[0] == 1
        assert vectors._conn.execute("SELECT count(*) FROM lcm_chunk_vectors").fetchone()[0] > 0
    assert store.expected(q)["store_text"] == variant
    assert json.loads((store.directory(q) / "speakers.json").read_text()) == {"1": "Ada", "2": "Bo", "3": "Ada"}


def test_lme_byte_identical_across_variants(tmp_path, lme, monkeypatch):
    q = datasets.normalize_lme({"question_id": "synthetic-lme", "question_type": "multi-session",
        "question": "teal?", "answer": "yes", "question_date": "2025-01-03",
        "haystack_session_ids": ["one"], "haystack_dates": ["2025-01-01"],
        "haystack_sessions": [[{"role": "user", "content": "Ada: teal key"}]]}, "lme-s", lme)
    snapshots = []
    for variant in VARIANTS:
        store = make_store(tmp_path, lme, monkeypatch, variant)
        store.build(q)
        snapshots.append([(store.directory(q) / name).read_bytes() for name in ("lcm.db", "speakers.json")])
    base = make_store(tmp_path / "base", lme, monkeypatch)
    base.build(q)
    assert all(s == [(base.directory(q) / n).read_bytes() for n in ("lcm.db", "speakers.json")] for s in snapshots)
    with base.engine(q) as (engine, _):
        monkeypatch.setattr(delivery, "dispatch", lambda *a: '{"hits":[{"store_id":1}]}')
        assert delivery.single(q, engine, hit_label="role", store_text="name-date")[0].endswith("user: Ada: teal key")


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("label", ("speaker", "role"))
def test_single_labels_once(tmp_path, lme, monkeypatch, variant, label):
    q = conversations()[0]
    store = make_store(tmp_path, lme, monkeypatch, variant)
    store.build(q)
    speakers = {"1": "Ada", "2": "Bo"} if label == "speaker" else {"1": "WRONG", "2": "WRONG"}
    monkeypatch.setattr(delivery, "dispatch", lambda *a: '{"hits":[{"store_id":1},{"store_id":2}]}')
    with store.engine(q) as (engine, _):
        evidence, _ = delivery.single(q, engine, speakers, hit_label=label, store_text=variant)
    assert evidence.splitlines() == [
        f"[2025-01-01T22:30:00+00:00] {'Ada' if label == 'speaker' else 'user'}: teal key 0",
        f"[2025-01-01T22:30:00+00:00] {'Bo' if label == 'speaker' else 'assistant'}: Ada: keep this literal prefix"]


def test_store_variants_never_reused(tmp_path, lme, monkeypatch):
    q = conversations()[0]
    built = []
    for variant in VARIANTS:
        store = make_store(tmp_path, lme, monkeypatch, variant)
        assert store.build(q) == "built"
        assert store.build(q) == "reused"
        built.append(store)
    assert len({s.generation for s in built}) == 3
    for store in built:
        for other in built:
            if store is other:
                continue
            with pytest.raises(RuntimeError, match="manifest mismatch"):
                # Force a cross-variant directory to prove the open guard, not only the path hash.
                monkeypatch.setattr(other, "directory", store.directory)
                with other.engine(q):
                    pass
            monkeypatch.setattr(other, "directory", stores.Stores.directory.__get__(other))


def cli(tmp_path, command="answer", *extra):
    return run.parser().parse_args([command, "--split-label", "synthetic", "--store-root", str(tmp_path / "stores"),
                                   "--output", str(tmp_path / "run"), *extra])


@pytest.mark.parametrize("variant", VARIANTS)
def test_mcp_reopens_matching_variant(tmp_path, lme, monkeypatch, variant):
    q = conversations()[0]
    store = make_store(tmp_path / "stores", lme, monkeypatch, variant)
    store.build(q)
    args = cli(tmp_path, "answer", "--store-text", variant, "--summaries", "deterministic")
    spec = {**run.server_spec(args, "synthetic-data", q), "log": str(tmp_path / "mcp.jsonl")}
    assert spec["store_text"] == variant and spec["hit_label"] == "n/a"
    out = io.StringIO()
    request = {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
               "params": {"name": "lcm_grep", "arguments": {"query": "teal"}}}
    mcp_reader.serve(spec, io.StringIO(json.dumps(request) + "\n"), out)
    reply = json.loads(out.getvalue())["result"]
    assert not reply["isError"] and "teal" in reply["content"][0]["text"]


@pytest.mark.parametrize("mode,expected_label", [("single", "role"), ("agentic", "n/a"), ("full_context", "n/a")])
def test_run_records_switches_and_passes_delivery(tmp_path, lme, monkeypatch, mode, expected_label):
    q = conversations()[0]
    args = cli(tmp_path, "answer", "--store-text", "plain", "--hit-label", "role", "--delivery", mode)
    args.output.mkdir()
    store = make_store(args.store_root, lme, monkeypatch, "plain")
    store.build(q)
    seen = []
    def answer(*a, **kw):
        seen.append(kw)
        return {"status": "ok", "hypothesis": "synthetic"}
    monkeypatch.setattr(delivery, "answer", answer)
    monkeypatch.setattr(run, "versions", lambda: {})
    if mode == "single":
        # A role-labelled run must not even open the speaker sidecar.
        (store.directory(q) / "speakers.json").write_text("not JSON")
    run.run_jobs([run.run(args, None, [q], "synthetic-data", "ids", {}, store)], 1)
    manifest = json.loads((args.output / "run-manifest.json").read_text())
    assert manifest["store_text"] == "plain" and manifest["hit_label"] == expected_label
    if mode == "single":
        assert seen[0]["hit_label"] == "role" and seen[0]["store_text"] == "plain"
    else:
        assert seen


@pytest.mark.parametrize("key,value", [("store_text", "plain"), ("hit_label", "role")])
def test_judge_refuses_switch_mismatch(tmp_path, monkeypatch, key, value):
    args = cli(tmp_path, "judge", "--delivery", "single")
    args.output.mkdir()
    q = SimpleNamespace(qid="0")
    monkeypatch.setattr(run, "commit", lambda _: BASE)
    manifest = {"dataset_sha256": "d", "id_list_sha256": "i", "split_label": "synthetic", "question_ids": ["0"],
        "prompts_sha256": run.prompt_hashes(), "harness_commit": BASE, "lcmx_commit": BASE,
        "harness_tree_sha256": common.sha(b"".join(p.read_bytes() for p in sorted(Path(run.__file__).parent.glob("*.py")))),
        "store_text": "speaker-prefix", "hit_label": "speaker"}
    manifest[key] = value
    common.write_json(args.output / "run-manifest.json", manifest)
    with pytest.raises(ValueError, match="identity mismatch"):
        next(run.run(args, None, [q], "d", "i", {}, None))


def test_score_995_arms_and_identity(tmp_path):
    variants = [("plain", "role"), ("plain", "speaker"), ("name-date", "speaker")]
    runs = [arm(tmp_path, f"A{i}", COMMITS[0], [True], store_text=text, hit_label=label)
            for i, (text, label) in enumerate(variants)]
    result = score.score(runs, tmp_path / "score")
    assert len(result["arms"]) == len(result["paired"]) == 3
    assert {"store_text", "hit_label"} <= set(score.IDENTITY)
    common.write_json(runs[1] / "run-manifest.json", {**json.loads((runs[1] / "run-manifest.json").read_text()),
                                                    "dataset_sha256": "different"})
    with pytest.raises(ValueError, match="identical dataset_sha256"):
        score.score(runs, tmp_path / "mismatch")


def test_cross_commit_pairs_preserve_switch_identity(tmp_path):
    runs = [arm(tmp_path, f"{i}-{j}", commit, [True], store_text=text, hit_label=label)
            for i, commit in enumerate(COMMITS)
            for j, (text, label) in enumerate((("plain", "role"), ("plain", "speaker"), ("name-date", "speaker")))]
    result = score.score(runs, tmp_path / "score", pair_across_commits=True)
    assert len(result["arms"]) == 6 and len(result["paired"]) == 3


def test_switches_reach_real_single_prompt(tmp_path, lme, monkeypatch):
    q = conversations()[0]
    store = make_store(tmp_path, lme, monkeypatch, "name-date")
    store.build(q)
    seen = []
    def call(reader, messages, records, **kw):
        seen.append(messages[1]["content"])
        records.append({"status": "ok", "input_tokens": 10})
        return {"content": "drawer"}
    monkeypatch.setattr(delivery, "dispatch", lambda *a: '{"hits":[{"store_id":1}]}')
    with store.engine(q) as (engine, _):
        result = delivery.answer(q, "single", SimpleNamespace(call=call), models.READERS[0], engine,
                                 hit_label="role", store_text="name-date")
    assert result["status"] == "ok" and "user: teal key 0" in seen[0] and "Ada" not in seen[0]
