"""#1718: sibling media shapes retain captions without previewing scaffolding."""

import base64
import json
from pathlib import Path
import time

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
