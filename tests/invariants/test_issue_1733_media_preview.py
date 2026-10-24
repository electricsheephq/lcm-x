"""Preview-only regressions from #1733, #1727 and #1734."""

import json
import re
import tracemalloc

import pytest

from hermes_lcm.externalize import _build_externalized_placeholder, _media_payload_preview_content
from hermes_lcm.externalize import _raw_payload_preview, maybe_externalize_payload
from hermes_lcm.ingest_content import _externalization_kind_for_message


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


# Review reproducers are copied here, rather than imported from operator artifacts.
PROSE = [
    "[[Daily Note]] deployment failed on the zebracluster ",
    "[[Daily Note]] don't deploy, zebracluster failed ",
    "('see screenshot') deployment failed on the zebracluster ",
    '("urgent") deployment failed on the zebracluster ',
    "((note)) deployment failed on the zebracluster ",
    "({draft}) deployment failed on the zebracluster ",
    "([1]) deployment failed on the zebracluster ",
    "[[1](http://x)] deployment failed ",
    '[] ["urgent"] deployment failed on the zebracluster ',
    "[] ('see image') deployment failed ",
    "[] [[Note]] deployment failed ",
    "{}\nnotes: deployment failed ",
    "[] [see attached] deployment failed ",
    '[] "quoted" deployment failed ',
    "(see screenshot) deployment failed ",
    "('urgent') deployment failed ",
    "[[Incident-42]] deployment failed ",
    "([link](https://x.y)) deployment failed ",
    "((see attached)) deployment failed ",
    "[] ('urgent') deployment failed ",
    "{} [[Incident-42]] deployment failed ",
    '("quoted") deployment failed ',
    "({tag}) deployment failed ",
    "(urgent) deployment failed ",
    "[link](https://x.y) deployment failed ",
    "[] deployment failed ",
    "[] [{}] deployment failed ",
    "[[_TOC_]]\n\nThe zebracluster deploy failed, screenshot below:\n",
    "[[Daily Note]] deployment failed on the zebracluster, screenshot: ",
    "('urgent') deploy failed on the zebracluster ",
    "((see attached)) deploy failed on the zebracluster ",
    "deploy failed on the zebracluster ",
]
BRACE_PROSE = [
    "{}\nnotes: deployment failed {config} ",
    "{1} deployment failed on {2} ",
    "{ (see image) deployment failed } ",
    "{True story} deployment failed ",
    "{} {1} notes ",
    "{None story} deployment failed ",
    "{False story} deployment failed ",
    "{1 thing to note ",
    "{(see image) ",
    "{1} deployment failed on cluster {2} ",
]


@pytest.mark.parametrize("prefix,suffix", [(p, "") for p in PROSE] + [(p, " }") for p in BRACE_PROSE])
def test_review_prose_function_and_externalization(tmp_path, prefix, suffix):
    # Same inline media shape as the real ingest classifier and externalization caller.
    content = prefix + "data:image/png;base64," + "QUJD" * 14_000 + suffix
    expected = (prefix + suffix).strip()
    assert _media_payload_preview_content(content) == expected
    assert _externalization_kind_for_message({"role": "user", "content": content}) == "media_payload"

    class Config:
        large_output_externalization_enabled = True
        large_output_externalization_threshold_chars = 12_000
        large_output_externalization_dir = str(tmp_path / "payloads")

    result = maybe_externalize_payload(
        content, kind="media_payload", role="user", session_id="fixture", config=Config(),
        hermes_home=str(tmp_path), write=False, create_storage=False,
    )
    assert result is not None
    assert _raw_payload_preview(expected, 300).split(" ... ")[0] in result["placeholder"]
    assert "data:" not in result["placeholder"] and "QUJD" not in result["placeholder"]


@pytest.mark.parametrize("content", [
    "(b'oops' typed by user) deployment failed data:image/png;base64,QUJD",
    "(('deployment failed', b'\\x89') broken",
    "[[deployment failed \\x89 broken",
    "{} {'broken': b'\\x89'",
    "[] [b'broken deployment failed",
])
def test_failed_repr_with_bytes_stays_opaque(content):
    assert_preview(content, "")


@pytest.mark.parametrize("length", [49_000, 53_600])
@pytest.mark.parametrize("shape", ["text-part", "caption", "bare-text", "apostrophe"])
def test_review_long_text_literal_keeps_base_preview(length, shape):
    words = ("deployment failed on the zebracluster after the upgrade, see logs. " * 800)[:length]
    if shape == "apostrophe":
        words = "it's " + words
    if shape == "caption":
        value = PART | {"caption": words}
    elif shape == "bare-text":
        value = [words, b"\x89PNG"]
    else:
        value = [{"type": "text", "text": words}, PART | {"caption": ""}]
    content = repr(value)
    preview = _media_payload_preview_content(content)
    assert _raw_payload_preview(preview, 300) == _raw_payload_preview(words.strip(), 300)
    # A second stub with the same metadata isolates byte-for-byte preview parity.
    assert _build_externalized_placeholder(SUMMARY, content=content) == _build_externalized_placeholder(
        SUMMARY, content=json.dumps([{"type": "text", "text": words}]),
    )


@pytest.mark.parametrize("text", [
    "deployment failed \\ in log\n" * 3000,
    "deployment failed 🙂 漢字\t" * 3000,
    "deployment failed \\n \\u0027 'quoted' \"double\" " * 2000,
], ids=["backslashes", "unicode", "quotes"])
@pytest.mark.parametrize("ascii_only", [False, True])
def test_long_literal_slices_respect_escapes(text, ascii_only):
    literal = ascii(text) if ascii_only else repr(text)
    content = "[" + literal + ", b'\\x89']"
    preview = _media_payload_preview_content(content)
    assert _raw_payload_preview(preview, 300) == _raw_payload_preview(text.strip(), 300)
    # Count decoded text, not repr source characters, when sizing the bounded windows.
    assert len(preview) >= min(len(text.strip()), 2 * 512 * 8)


@pytest.mark.parametrize("value", [
    [[PART]], ([PART],), [[CAPTION, b"\x89PNG"]], ({CAPTION, b"\x89PNG"},),
    {(1, CAPTION)},
])
def test_review_nested_sequence_captions(value):
    assert_preview(repr(value), CAPTION)


def test_review_bytes_keys_do_not_collapse_values():
    content = repr({b"part1": {"caption": CAPTION}, b"part2": {"caption": "second"}, b"img": b"\x89PNG"})
    assert_preview(content, CAPTION + "\nsecond")


@pytest.mark.parametrize("content", [
    "{'type': 'image', 'data': '" + "QUJD" * 30,
    '["deployment failed", {"audio": "' + "QUJD" * 30,
    "{'k': " + "é" * 20 + "b'opaque}",
])
def test_failed_repr_fallback_does_not_expose_cut_media(content):
    assert_preview(content, "")


def test_nested_sequence_drops_anonymous_encoded_media_but_keeps_caption():
    assert_preview(json.dumps([["QUJD" * 30, CAPTION]]), CAPTION)
