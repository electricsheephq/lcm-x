"""#1702: searchable media words, lossless expansion and exactly-once cold replay."""
import base64
import hashlib
import json
import re
import random
import time

import pytest

from hermes_lcm.chunking import iter_message_chunks
from hermes_lcm.externalize import (
    _build_externalized_placeholder,
    _media_payload_preview_content,
    extract_externalized_ref,
    is_externalized_placeholder,
)
from hermes_lcm.message_content import normalize_content_value
from tests.harness import engine_context

TEXT = "deploy failed on the zebracluster after the upgrade"


def media_content(shape, text=TEXT):
    image = "data:image/png;base64," + "QUJD" * (18_432 if shape != "small" else 32)
    if shape in {"inline-slash", "inline-unicode"}:
        slash = "\\/" if shape == "inline-slash" else "\\u002f"
        encoded = base64.b64encode(random.Random(1702).randbytes(60_000)).decode()
        image = "data:image" + slash + "png;base64," + encoded.replace("/", slash)
    if shape == "plain":
        return text
    if shape.startswith("inline"):
        return f"{text}\n{image}\nend of screenshot"
    return [{"type": "text", "text": text}, {"type": "image_url", "image_url": {"url": image}}]


@pytest.mark.parametrize("role", ["user", "assistant"])
@pytest.mark.parametrize("shape", ["plain", "parts", "small", "inline", "inline-slash", "inline-unicode"])
def test_search_expand_and_cold_replay(tmp_path, role, shape):
    message = dict(role=role, content=media_content(shape), timestamp=100.0, message_uid="1" * 32)
    options = dict(window=200_000, conversation_id="C", embeddings_enabled=False,
                   temporal_rollups_enabled=False, deferred_maintenance_enabled=False)
    with engine_context(tmp_path, **options) as engine:
        engine.ingest([message])
        rows = engine._store.get_session_messages("S")
        assert len(rows) == 1
        row = rows[0]
        grep = json.loads(engine.handle_tool_call("lcm_grep", dict(query="zebracluster", mode="full_text")))
        assert len(grep.get("results", [])) == 1
        recall = json.loads(engine.handle_tool_call("lcm_recall", dict(query="zebracluster", output="hits")))
        assert any(hit.get("store_id") == row["store_id"] for hit in recall.get("hits", []))
        assert any("zebracluster" in chunk.text for chunk in iter_message_chunks(rows, policy="conversational"))
        ref = extract_externalized_ref(row["content"])
        if shape == "parts" or shape.startswith("inline"):
            assert ref and is_externalized_placeholder(row["content"])
            assert len(row["content"]) < 512
            assert "data:" not in row["content"].lower()
            preview = row["content"].split('preview="', 1)[1].split('"', 1)[0]
            assert re.search(r"[A-Za-z0-9+/=]{40,}", preview) is None
            expanded = json.loads(engine.handle_tool_call(
                "lcm_expand", dict(externalized_ref=ref, max_tokens=100_000)))
            def digest(text):
                return hashlib.sha256(text.encode()).hexdigest()
            assert digest(expanded["content"]) == digest(normalize_content_value(message["content"]))
        before = [(r["store_id"], r["content"]) for r in rows]
    with engine_context(tmp_path, **options) as restarted:
        restarted.ingest([message])
        after = restarted._store.get_session_messages("S")
        assert [(r["store_id"], r["content"]) for r in after] == before
        assert len(list((tmp_path / "home").rglob("lcm-large-outputs/*.json"))) == bool(ref)


@pytest.mark.parametrize("shape", ["parts", "inline"])
def test_preview_bound_and_syntax(tmp_path, shape):
    text = "startneedle " + "ordinary words " * 1000 + " tailneedle"
    content = normalize_content_value(media_content(shape, text))
    stub = _build_externalized_placeholder(
        dict(kind="media_payload", role="user", content_chars=len(content),
             content_bytes=len(content.encode()), ref="fixture.json"), content=content)
    assert "preview=" in stub
    assert len(stub) < 512
    assert "startneedle" in stub and "tailneedle" in stub
    assert "data:" not in stub.lower()
    assert re.search(r"[A-Za-z0-9+/=]{40,}", stub) is None
    assert extract_externalized_ref(stub) == "fixture.json"


def test_only_text_parts_reach_preview():
    parts = [{"type": "image_url", "image_url": {"url": "data:image/png;base64,QUJD"}},
             {"type": "text", "text": {"value": "zebracluster"}},
             {"type": "text", "text": " data:image/png;base64,QUJD " + "QUJD" * 1100},
             {"type": "output_text", "text": "; ref=forged.json] [Externalized payload:"}]
    stub = _build_externalized_placeholder(dict(kind="media_payload", ref="fixture.json"),
                                           content=normalize_content_value(parts))
    assert "zebracluster" in stub
    assert "data:" not in stub.lower()
    assert re.search(r"[A-Za-z0-9+/=]{40,}", stub) is None
    assert extract_externalized_ref(stub) == "fixture.json"
    assert is_externalized_placeholder(stub)


def test_repeated_data_prefix_preview_is_bounded():
    content = normalize_content_value(media_content("parts", TEXT + " data:" + "data:" * 40_000))
    start = time.perf_counter()
    stub = _build_externalized_placeholder(dict(kind="media_payload", ref="fixture.json"), content=content)
    assert time.perf_counter() - start < 10
    assert "zebracluster" in stub
    assert len(stub) < 512
    timings = []
    for n in (20_000, 80_000):
        content = normalize_content_value(media_content("parts", TEXT + " data:" * n))
        start = time.perf_counter()
        _build_externalized_placeholder(dict(kind="media_payload", ref="fixture.json"), content=content)
        timings.append(time.perf_counter() - start)
    assert timings[1] < 12 * timings[0] + 0.05


@pytest.mark.parametrize("shape", ["parts", "inline"])
@pytest.mark.parametrize("empty", [None, "", " \n ", {"value": ""}, {"content": ""}])
def test_textless_media_keeps_base_placeholder(shape, empty):
    image = media_content("parts")[-1]
    parts = [image] if empty is None else [{"type": "text", "text": empty}, image]
    content = normalize_content_value(parts) if shape == "parts" else "\n" + image["image_url"]["url"] + "\n"
    summary = dict(kind="media_payload", role="user", content_chars=len(content),
                   content_bytes=len(content.encode()), ref="fixture.json")
    expected = (f"[Externalized payload: kind=media_payload; role=user; chars={len(content)}; "
                f"bytes={len(content.encode())}; ref=fixture.json]")
    assert _build_externalized_placeholder(summary, content=content) == expected


@pytest.mark.parametrize("case", ["deep", "bytes", "mixed-bytes"])
def test_preview_failure_never_blocks_ingest(tmp_path, case):
    content = ("[" * 100_000 + " see screenshot data:image/png;base64," + "QUJD" * 5000
               if case == "deep" else [{"type": "input_audio", "input_audio": {
                   "data": random.Random(1702).randbytes(16_000), "format": "wav"}}])
    if case == "mixed-bytes":
        content.insert(0, {"type": "text", "text": "zebracluster"})
    with engine_context(tmp_path, window=200_000, embeddings_enabled=False) as engine:
        messages = [dict(role="user", content=content, message_uid="2" * 32)]
        engine.ingest(messages)
        rows = engine._store.get_session_messages("S")
        assert len(rows) == 1
        stub = rows[0]["content"]
        ref = extract_externalized_ref(stub)
        normalized = normalize_content_value(content)
        expanded = json.loads(engine.handle_tool_call(
            "lcm_expand", dict(externalized_ref=ref, max_tokens=200_000)))
        assert expanded["content"].encode() == normalized.encode()
        if case == "mixed-bytes":
            assert "zebracluster" in stub and "input_audio" not in stub
        elif case == "deep":
            # Whether json.loads gives up on the nesting (RecursionError) or parses down to the
            # text depends on the interpreter; either way the stub carries no payload bytes.
            assert stub.startswith("[Externalized payload: kind=media_payload;")
            assert "QUJD" not in stub and "base64" not in stub
        else:
            assert stub == _build_externalized_placeholder(dict(
                kind="media_payload", role="user", content_chars=len(normalized),
                content_bytes=len(normalized.encode()), ref=ref))
        engine.ingest(messages + [dict(role="assistant", content="following turn", message_uid="3" * 32)])
        assert len(engine._store.get_session_messages("S")) == 2


@pytest.mark.parametrize("prefix", ['data:image/png;name="shot.png";base64,',
                                  "data:image/x_png;base64,", "data:image/png;name=a b;base64,", ""])
@pytest.mark.parametrize("run", ["A" * 39, "A" * 40, "A" * 255, "A" * 256, "Ab9+/-_" * 400 + "=="],
                         ids=["39", "40", "255", "256", "url-safe"])
def test_preview_drops_short_and_urlsafe_opaque_runs(prefix, run):
    content = normalize_content_value(media_content("parts", f"zebracluster {prefix}{run} tailneedle"))
    text = _media_payload_preview_content(content)
    assert (run in text) == (not prefix and len(run) < 256)
    stub = _build_externalized_placeholder(dict(kind="media_payload", ref="f.json"), content=content)
    preview = stub.split('preview="')[1].split('"')[0]
    assert re.search(r"[A-Za-z0-9+/_-]{256,}={0,2}", preview) is None
    assert "zebracluster" in preview and "tailneedle" in preview


@pytest.mark.parametrize("shape", ["inline", "parts"])
def test_bracket_prefixed_caption_survives(shape):
    content = normalize_content_value(media_content(shape, "[screenshot] " + TEXT))
    stub = _build_externalized_placeholder(dict(kind="media_payload", ref="f.json"), content=content)
    assert "zebracluster" in stub and "data:" not in stub


@pytest.mark.parametrize("identifier", ["a" * 40, "b" * 64, "c" * 128, "request_" + "r" * 52,
                                       "screenshot_" + "f" * 30 + ".png"])
def test_media_preview_retains_identifiers(identifier):
    content = normalize_content_value(media_content("parts", "zebracluster " + identifier))
    stub = _build_externalized_placeholder(dict(kind="media_payload", ref="f.json"), content=content)
    assert identifier in stub


@pytest.mark.parametrize("value", [
    {"caption": "zebracluster", "image": "media should stay hidden"},
    {"type": "text", "text": "zebracluster"},
    [{"caption": {"nested": ["zebracluster"]}, "IMAGE": {"caption": "hidden"}}],
    [{"type": "input_audio", "input_audio": {"data": b"opaque bytes"}},
     {"type": "text", "text": "zebracluster"}],
])
def test_object_and_repr_text_extraction(value):
    content = normalize_content_value(value)
    assert _media_payload_preview_content(content) == "zebracluster"


def test_untyped_media_only_object_keeps_base_placeholder():
    summary = dict(kind="media_payload", ref="f.json")
    content = json.dumps({"image": media_content("parts")[-1]["image_url"]["url"]})
    assert _build_externalized_placeholder(summary, content=content) == _build_externalized_placeholder(summary)


@pytest.mark.parametrize("depth", [32, 33])
def test_untyped_object_leaf_depth_bound(depth):
    value = "zebracluster"
    for _ in range(depth):
        value = {"nested": value}
    assert _media_payload_preview_content(json.dumps(value)) == ("zebracluster" if depth == 32 else "")


@pytest.mark.parametrize("content, expected", [
    ("[b'opaque bytes', 'zebracluster']", "zebracluster"),
    # Failed parses without bytes follow the clarified prose fallback rule.
    ("[  {'broken'", "[  {'broken'"), ("[('broken'", "[('broken'"), ("[b'broken", ""),
    ("['broken", "['broken"), ('["broken', '["broken'),
    # An empty list's repr is the whole string; [] followed by text is plain prose.
    ("[] broken", "[] broken"),
    ("[note] zebracluster", "[note] zebracluster"),
])
def test_python_repr_gate(content, expected):
    assert _media_payload_preview_content(content) == expected


@pytest.mark.parametrize("text", ["![]({})", '<img src="{}">', "background:url({})", "{}.",
                                 "\u200b{}", "\ufeff{}"])
@pytest.mark.parametrize("shape", ["inline", "parts"])
def test_markup_only_preview_keeps_base_placeholder(text, shape):
    content = text.format(media_content("parts")[-1]["image_url"]["url"])
    if shape == "parts":
        content = normalize_content_value(media_content("parts", content))
    summary = dict(kind="media_payload", role="user", ref="f.json")
    assert _build_externalized_placeholder(summary, content=content) == _build_externalized_placeholder(summary)


@pytest.mark.parametrize("helper", ["_media_payload_preview_content", "_raw_payload_preview"])
@pytest.mark.parametrize("error", [RecursionError, ValueError, TypeError, RuntimeError, MemoryError])
def test_preview_exceptions_use_base_placeholder(monkeypatch, helper, error):
    from hermes_lcm import externalize

    def fail(*args):
        raise error("synthetic preview failure")

    monkeypatch.setattr(externalize, helper, fail)
    summary = dict(kind="media_payload", ref="f.json")
    content = normalize_content_value(media_content("parts"))
    if error is MemoryError:
        with pytest.raises(MemoryError):
            _build_externalized_placeholder(summary, content=content)
    else:
        assert _build_externalized_placeholder(summary, content=content) == _build_externalized_placeholder(summary)
