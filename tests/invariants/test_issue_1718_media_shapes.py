"""#1718: sibling media shapes retain captions without previewing scaffolding."""

import base64
import json
from pathlib import Path
import time
import tracemalloc

import pytest

from hermes_lcm.externalize import (
    _build_externalized_placeholder,
    _media_payload_preview_content,
    extract_externalized_ref,
    is_externalized_placeholder,
)
from hermes_lcm.message_content import normalize_content_value
from hermes_lcm.sanitize import _sanitize_active_assistant_content
from tests.invariants.test_issue_1702_media_text import media_content


CAPTION = "deployment failed on the zebracluster"
SUMMARY = dict(kind="media_payload", role="user", ref="fixture.json")
DESCRIPTOR_KEYS = (
    "mime_type", "mimetype", "mime", "content_type", "content-type", "contenttype",
    "media_type", "media-type", "mediatype", "format", "encoding", "detail",
)


def assert_caption_preview(content):
    assert _media_payload_preview_content(content) == CAPTION
    stub = _build_externalized_placeholder(SUMMARY, content=content)
    assert CAPTION in stub
    assert len(stub) < 512
    assert stub.endswith("; ref=fixture.json]")
    assert extract_externalized_ref(stub) == "fixture.json"
    assert is_externalized_placeholder(stub)
    return stub


@pytest.mark.parametrize("part_type", ["text", "input_text", "output_text"])
@pytest.mark.parametrize("outer", ["text", "content", "value"])
@pytest.mark.parametrize("inner", [None, "value", "content", "text"])
def test_nested_typed_text_matches_assistant_cleanup(part_type, outer, inner):
    value = CAPTION if inner is None else {inner: CAPTION}
    part = {"type": part_type, outer: value}
    assert _sanitize_active_assistant_content([part]) == [part]
    content = normalize_content_value([part, media_content("parts")[-1]])
    assert_caption_preview(content)


@pytest.mark.parametrize("media_key", ["image", "IMAGE", "input_image", "audio", "video_url", "data"])
def test_non_json_dict_repr_filters_media(media_key):
    value = {media_key: b"opaque media bytes", "caption": CAPTION}
    content = normalize_content_value(value)
    assert content == str(value)
    assert_caption_preview(content)


@pytest.mark.parametrize("first_key", [0, (1, 2), None, 1.5, True])
def test_non_json_dict_repr_with_non_string_first_key(first_key):
    value = {first_key: b"opaque media bytes", "image": b"opaque media bytes", "caption": CAPTION}
    content = normalize_content_value(value)
    assert content == str(value)
    assert_caption_preview(content)


@pytest.mark.parametrize("prefix", ["{} ", "{  } ", "{'broken ", '{"broken ', "{b'broken "])
def test_dict_like_caption_gate(prefix):
    content = prefix + CAPTION + "\ndata:image/png;base64,QUJD\n" + "QUJD" * 100
    expected = prefix + CAPTION if prefix.replace(" ", "") == "{}" else ""
    assert _media_payload_preview_content(content) == expected
    stub = _build_externalized_placeholder(SUMMARY, content=content)
    assert "data:" not in stub and "QUJD" not in stub
    assert (CAPTION in stub) == bool(expected)


@pytest.mark.parametrize("key", DESCRIPTOR_KEYS)
def test_media_descriptor_metadata_is_not_caption(key):
    content = normalize_content_value({key.upper(): "image/png", "data": "QUJD" * 100})
    assert _media_payload_preview_content(content) == ""
    assert _build_externalized_placeholder(SUMMARY, content=content) == _build_externalized_placeholder(SUMMARY)


@pytest.mark.parametrize("key", ["caption", "alt", "description", "title", "name", "text", "content"])
def test_free_text_leaves_still_count(key):
    content = normalize_content_value({"mime_type": "image/png", "data": b"opaque media bytes", key: CAPTION})
    assert_caption_preview(content)


@pytest.mark.parametrize("prefix", ["[] ", "[  {'broken' ", "[('broken' ", "[b'broken ", "['broken ", '["broken '])
def test_list_like_caption_gate_preserves_truncated_repr_fallback(prefix):
    encoded = base64.b64encode(bytes(range(256)) * 20).decode("ascii")
    wrapped = "\n".join(encoded[i:i + 76] for i in range(0, len(encoded), 76))
    content = prefix + CAPTION + "\ndata:image/png;base64,QUJD\n" + "QUJD" * 100 + "\n" + wrapped
    expected = prefix + CAPTION if prefix == "[] " else ""
    assert _media_payload_preview_content(content) == expected
    stub = _build_externalized_placeholder(SUMMARY, content=content)
    if expected:
        assert CAPTION in stub
    else:
        assert stub == _build_externalized_placeholder(SUMMARY)
    assert "data:" not in stub and "QUJD" not in stub
    assert len(stub) < 512 and stub.endswith("; ref=fixture.json]")


@pytest.mark.parametrize("case", ["deep", "huge"])
def test_bounded_dict_repr_keeps_base_placeholder(case):
    if case == "deep":
        content = "{'nested':" * 10_000 + "{'image': b'opaque media bytes'}" + "}" * 10_000
    else:
        content = "{'image': b'" + "QUJD" * 300_000 + "'}"
    start = time.monotonic()
    assert _build_externalized_placeholder(SUMMARY, content=content) == _build_externalized_placeholder(SUMMARY)
    assert time.monotonic() - start < 5


@pytest.mark.parametrize("kind", ["raw_payload", "tool_result"])
@pytest.mark.parametrize("shape", ["plain", "parts", "small", "inline", "inline-slash", "inline-unicode"])
def test_nonmedia_placeholders_byte_identical_to_base(kind, shape):
    # Captured on e6fc9520efa8645ec0c8171b055b1fe389372c64 using these existing fixtures.
    expected = json.loads(Path(__file__).with_name("issue_1718_nonmedia_base.json").read_text())
    content = normalize_content_value(media_content(shape))
    summary = dict(SUMMARY, kind=kind, content_chars=len(content), content_bytes=len(content.encode()))
    actual = _build_externalized_placeholder(summary, content=content)
    assert actual.encode() == expected[kind][shape].encode()


def test_brace_wrapped_prose_keeps_its_caption():
    # Only a dict proves a brace-spanned string was a repr; prose in braces stays text, as on main.
    content = "{" + CAPTION + " data:image/png;base64," + "QUJD" * 100 + "}"
    preview = _media_payload_preview_content(content)
    assert CAPTION in preview and "QUJD" not in preview
    stub = _build_externalized_placeholder(SUMMARY, content=content)
    assert CAPTION in stub and "data:" not in stub and "QUJD" not in stub


@pytest.mark.parametrize("container", ["list", "dict"])
def test_repr_over_one_mib_keeps_its_caption(container):
    blob = bytes(range(256)) * 4_096  # 1 MiB of bytes; its repr is several MiB
    if container == "list":
        value = [{"type": "text", "text": CAPTION}, {"type": "image", "image": blob}]
    else:
        value = {"image": blob, "caption": CAPTION}
    content = normalize_content_value(value)
    assert content == str(value) and len(content) > 1_048_576
    start = time.monotonic()
    assert_caption_preview(content)
    assert time.monotonic() - start < 5


def test_data_envelope_is_walked():
    content = normalize_content_value(
        {"data": {"caption": CAPTION, "blob": "data:image/png;base64," + "QUJD" * 100}}
    )
    assert_caption_preview(content)


@pytest.mark.parametrize("content", [
    "{0: b'" + "\\x89PNG" * 64 + "', 'caption': 'x'}",  # non-string first key: not repr_like by prefix
    "{" + CAPTION + " " + "\\x89PNG" * 64 + "}",
])
def test_oversized_brace_span_is_never_previewed_unparsed(monkeypatch, content):
    import hermes_lcm.externalize as externalize

    # raising=False keeps the RED-on-base check behavioural (the old code has a fixed 64 MiB bound).
    monkeypatch.setattr(externalize, "_MEDIA_REPR_PARSE_LIMIT", 128, raising=False)
    assert len(content) > 128
    assert _media_payload_preview_content(content) == ""
    assert _build_externalized_placeholder(SUMMARY, content=content) == _build_externalized_placeholder(SUMMARY)


def test_typed_generic_dict_repr_keeps_its_caption():
    value = {"type": "deployment_error", "image": b"opaque media bytes", "caption": CAPTION}
    content = normalize_content_value(value)
    assert content == str(value)
    preview = _media_payload_preview_content(content)
    assert CAPTION in preview and "opaque" not in preview


@pytest.mark.parametrize("wrap", [tuple, set])
def test_tuple_and_set_captions_in_dict_repr(wrap):
    value = {"image": b"opaque media bytes", "caption": wrap([CAPTION])}
    content = normalize_content_value(value)
    assert content == str(value)
    assert_caption_preview(content)


@pytest.mark.parametrize("content", [
    "{'image': b'opaque', 'caption': '" + CAPTION + "', 'histogram': [0,0,0,0]}",
    "{'a': {'b': {'c': {'caption': '" + CAPTION + "'}}}, 'image': b'opaque'}",
])
def test_repr_element_limit_fails_closed(monkeypatch, content):
    import hermes_lcm.externalize as externalize

    # The base has no element bound; keep its RED result behavioural.
    monkeypatch.setattr(externalize, "_MEDIA_REPR_ELEMENT_LIMIT", 4, raising=False)
    assert content.count(",") + content.count(":") > 4
    assert _media_payload_preview_content(content) == ""
    assert _build_externalized_placeholder(SUMMARY, content=content) == _build_externalized_placeholder(SUMMARY)


def test_real_size_histogram_repr_has_bounded_peak_memory():
    content = "{'image': b'\\x89PNG', 'caption': '" + CAPTION + "', 'histogram': [" + "0," * 2_000_000 + "]}"
    tracemalloc.start()
    try:
        preview = _media_payload_preview_content(content)
        stub = _build_externalized_placeholder(SUMMARY, content=content)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    print(f"histogram repr peak_mib={peak / 1_048_576:.3f}")
    assert peak < 200 * 1_048_576
    assert preview == ""
    assert stub == _build_externalized_placeholder(SUMMARY)


@pytest.mark.parametrize("filler", [b"," * 30_000 + b":" * 30_000, "," * 30_000 + ":" * 30_000],
                         ids=["bytes", "str"])
def test_separators_inside_literals_are_not_elements(filler):
    # ~8 MiB of image bytes repr to ~65k "," and ":" characters; main kept this caption as text.
    # A str filler sits in a key: keys are never previewed, so the preview is the caption alone.
    value = {"image": filler, "caption": CAPTION} if isinstance(filler, bytes) else {
        "image": b"\x89PNG", "caption": CAPTION, filler: 0}
    content = str(value)
    assert content.count(",") + content.count(":") > 50_000
    assert_caption_preview(content)


def test_triple_quoted_strings_do_not_hide_elements(monkeypatch):
    import hermes_lcm.externalize as externalize

    monkeypatch.setattr(externalize, "_MEDIA_REPR_ELEMENT_LIMIT", 4, raising=False)
    # Pairing single quotes would read "', 0,0,...,0, '" as one string; the tokenizer does not.
    content = "[''' ''', " + "0," * 8 + " '']"
    assert _media_payload_preview_content(content) == ""


def test_untokenizable_repr_falls_back_to_counting_every_separator(monkeypatch):
    import hermes_lcm.externalize as externalize

    monkeypatch.setattr(externalize, "_MEDIA_REPR_ELEMENT_LIMIT", 4, raising=False)
    content = "{'image': b'opaque', 'caption': '" + CAPTION + "', 'n': [0,0,0,0,0], 'open': '"
    assert _media_payload_preview_content(content) == ""


@pytest.mark.parametrize("as_repr", [False, True])
@pytest.mark.parametrize("shape, expected", [
    ({"error": {"detail": CAPTION}, "image": "URI"}, CAPTION),
    ({"Content-Type": CAPTION, "image": "URI"}, ""),
    ({"msg": {"encoding": CAPTION}, "x": "URI"}, CAPTION),
    ({"format": CAPTION, "image_url": "URI"}, ""),
    ({"data": CAPTION, "image": "URI"}, ""),
])
def test_descriptor_leaves_are_scoped_to_media_dicts(as_repr, shape, expected):
    value = {k: "data:image/png;base64," + "QUJD" * 200 if v == "URI" else v for k, v in shape.items()}
    content = str(value) if as_repr else json.dumps(value, sort_keys=True)
    assert _media_payload_preview_content(content) == expected
    if expected:
        assert_caption_preview(content)
    else:
        assert _build_externalized_placeholder(SUMMARY, content=content) == _build_externalized_placeholder(SUMMARY)


def test_literal_eval_unhashable_set_is_treated_as_brace_prose():
    content = '{["x"]}'
    assert _media_payload_preview_content(content) == content
    assert 'preview="' in _build_externalized_placeholder(SUMMARY, content=content)


@pytest.mark.parametrize("container", ["dict", "list"])
def test_literal_eval_deep_unary_chain_keeps_base_placeholder(container):
    part = "{'caption': '" + CAPTION + "', 'n': " + "-" * 6000 + "1, 'image': b'\\x89PNG'}"
    content = part if container == "dict" else "[" + part + "]"
    assert _media_payload_preview_content(content) == ""
    assert _build_externalized_placeholder(SUMMARY, content=content) == _build_externalized_placeholder(SUMMARY)


@pytest.mark.parametrize("error", [TypeError, MemoryError])
def test_literal_eval_parser_errors_fail_closed(monkeypatch, error):
    import ast

    def fail(_content):
        raise error("parser failure")

    monkeypatch.setattr(ast, "literal_eval", fail)
    content = "{'image': b'opaque', 'caption': '" + CAPTION + "'}"
    assert _media_payload_preview_content(content) == ""
    assert _build_externalized_placeholder(SUMMARY, content=content) == _build_externalized_placeholder(SUMMARY)
