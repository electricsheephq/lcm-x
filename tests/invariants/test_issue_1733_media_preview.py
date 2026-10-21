"""Preview-only regressions from #1733, #1727 and #1734."""

import json
import re
import tracemalloc

import pytest

from hermes_lcm.externalize import _build_externalized_placeholder, _media_payload_preview_content


CAPTION = "deployment failed"
PART = {"type": "image", "image": b"\x89PNG\r\n", "caption": CAPTION}
SUMMARY = dict(kind="media_payload", role="user", ref="fixture.json")


def assert_preview(content, expected):
    preview = _media_payload_preview_content(content)
    assert "\\x" not in preview
    assert "b'" not in preview
    assert not re.search(r"[A-Za-z0-9+/_-]{64,}={0,2}", preview)
    assert preview == expected
    stub = _build_externalized_placeholder(SUMMARY, content=content)
    if expected:
        assert CAPTION in stub if CAPTION in expected else expected in stub
    else:
        assert stub == _build_externalized_placeholder(SUMMARY)


@pytest.mark.parametrize("content,expected", [
    (str([PART]), CAPTION),
    (json.dumps([{"type": "image", "image_url": "data:...", "caption": CAPTION}]), CAPTION),
    (str((PART,)), CAPTION),
    (str({"type": "image", "image": b"\x89PNG\r\n"}), ""),
    ("{0, b'opaque', 'deployment failed', 'data:image/png;base64,QUJD'}", CAPTION),
    ("[]\n" + str([{"type": "image", "image": b"\x89PNG\r\n"}, "caption words"]), "caption words"),
    ("[] " + str([b"\x89PNG\r\n"]), ""),
    ("{} " + str(PART), CAPTION),
], ids=["list-repr", "list-json", "tuple-repr", "typed-metadata", "numeric-set",
        "empty-list-list-repr", "empty-list-bytes-repr", "empty-dict-dict-repr"])
def test_issue_shapes(content, expected):
    assert_preview(content, expected)


@pytest.mark.parametrize("key", ["0", "None", "(1, 2)"])
def test_failed_dict_repr_is_not_prose(key):
    content = "{" + key + ": " + "[" * 250 + "0" + "]" * 250 + ", 'image': b'\\x89PNG'}"
    assert_preview(content, "")


@pytest.mark.parametrize("prefix", ["[] ", "[]\n", "{} ", "{  } "])
def test_empty_container_before_prose_stays_prose(prefix):
    assert_preview(prefix + CAPTION, prefix + CAPTION)


@pytest.mark.parametrize("wrapper", [list, tuple])
@pytest.mark.parametrize("field", ["caption", "alt", "title"])
def test_wrapped_caption_fields(wrapper, field):
    content = repr(wrapper([{"type": "image", "image": b"\x89PNG", field: CAPTION}]))
    assert_preview(content, CAPTION)


@pytest.mark.parametrize("count", [1, 2])
def test_large_bytes_caption_recovery_memory(count):
    # A 32 MiB aggregate stays below the unchanged 64 MiB source ceiling.
    part = "{'type': 'image', 'image': b'" + "x" * (32 * 1_048_576 // count) + "', 'caption': 'deployment failed'}"
    content = part if count == 1 else "[" + ", ".join([part] * count) + "]"
    tracemalloc.start()
    try:
        preview = _media_payload_preview_content(content)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    print(f"bytes_repr count={count} content_chars={len(content)} peak_bytes={peak}")
    assert peak <= 2 * len(content)
    assert preview == "\n".join([CAPTION] * count)
    assert "\\x" not in preview and "b'" not in preview
    assert not re.search(r"[A-Za-z0-9+/_-]{64,}", preview)


def test_two_32_mib_literals_still_obey_source_ceiling():
    part = "{'type': 'image', 'image': b'" + "x" * (32 * 1_048_576) + "', 'caption': 'deployment failed'}"
    content = "[" + part + ", " + part + "]"
    assert len(content) > 64 * 1_048_576
    assert_preview(content, "")
