"""#60: explicit-session externalized expansion is bounded and follows rotation lineage."""

import json

import pytest

from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from hermes_lcm.externalize import get_large_output_storage_dir, maybe_externalize_tool_output


def _rotation_engine(tmp_path, *, hermes_home=None):
    engine = LCMEngine(
        config=LCMConfig(
            database_path=str(tmp_path / "lcm.db"),
            large_output_externalization_enabled=True,
            large_output_externalization_threshold_chars=200,
        ),
        hermes_home=str(hermes_home or tmp_path / "hermes"),
    )
    engine.on_session_start("session-a", platform="discord", conversation_id="thread", context_length=200_000)
    return engine


def _rotate(engine, old, new):
    engine.on_session_start(
        new,
        platform="discord",
        conversation_id="thread",
        context_length=200_000,
        boundary_reason="compression",
        old_session_id=old,
    )
    assert engine.current_session_id == new


def _externalize(engine, session_id, content):
    result = maybe_externalize_tool_output(
        content,
        tool_call_id=f"call-{session_id}",
        session_id=session_id,
        config=engine._config,
        hermes_home=engine._hermes_home,
    )
    return result["path"].name


def _restart(engine, session_id="session-b"):
    engine.on_session_start(
        session_id, platform="discord", conversation_id="thread-2", context_length=200_000
    )


def _expand(engine, **args):
    return json.loads(engine.handle_tool_call("lcm_expand", args))


def test_explicit_session_reads_other_sessions_payload(tmp_path):
    engine = _rotation_engine(tmp_path)
    content = "RESULT in session a " * 50
    try:
        ref = _externalize(engine, "session-a", content)
        _restart(engine)
        assert "not found in current session" in _expand(engine, externalized_ref=ref)["error"]
        result = _expand(engine, externalized_ref=ref, session_id="session-a")
        assert result["content"] == content
        assert result["session_id"] == "session-a"
        assert result["has_more"] is False
    finally:
        engine.shutdown()


def test_explicit_session_payload_is_bounded_and_paged(tmp_path):
    engine = _rotation_engine(tmp_path)
    content = "RESULT in session a " * 1000
    try:
        ref = _externalize(engine, "session-a", content)
        _restart(engine)
        result = _expand(engine, externalized_ref=ref, session_id="session-a", max_tokens=200)
        assert result["content_truncated"] is True
        assert result["next_content_offset"] > 0
        pages = [result["content"]]
        while result["has_more"]:
            previous_offset = result["content_offset"]
            result = _expand(
                engine,
                externalized_ref=ref,
                session_id="session-a",
                max_tokens=200,
                content_offset=result["next_content_offset"],
            )
            assert result["content_offset"] > previous_offset
            pages.append(result["content"])
        assert "".join(pages) == content
    finally:
        engine.shutdown()


def test_wrong_explicit_session_is_refused(tmp_path):
    engine = _rotation_engine(tmp_path)
    try:
        ref = _externalize(engine, "session-a", "RESULT in session a " * 50)
        _restart(engine)
        result = _expand(engine, externalized_ref=ref, session_id="session-x")
        assert "not found in session session-x" in result["error"]
        assert "content" not in result
        for invalid in ("", 5):
            result = _expand(engine, externalized_ref=ref, session_id=invalid)
            assert result == {"error": "session_id must be a non-empty string"}
        result = _expand(engine, node_id=1, externalized_ref=ref, session_id="session-a")
        assert "Provide only one" in result["error"]
    finally:
        engine.shutdown()


def test_explicit_session_missing_payload_is_not_found(tmp_path):
    engine = _rotation_engine(tmp_path)
    try:
        ref = _externalize(engine, "session-a", "RESULT in session a " * 50)
        storage = get_large_output_storage_dir(
            engine._config, hermes_home=engine._hermes_home, create=False
        )
        (storage / ref).unlink()
        _restart(engine)
        assert _expand(engine, externalized_ref=ref, session_id="session-a") == {
            "error": f"Externalized payload {ref} not found in session session-a"
        }
    finally:
        engine.shutdown()


@pytest.mark.parametrize("path_mode", ["payload_symlink", "configured_directory"])
def test_explicit_session_rejects_cross_profile_payload_path(tmp_path, path_mode):
    engine = _rotation_engine(tmp_path)
    other_root = tmp_path / "other-profile"
    other_root.mkdir()
    other = _rotation_engine(other_root)
    try:
        ref = _externalize(other, "session-a", "RESULT from another profile " * 50)
        other_storage = get_large_output_storage_dir(
            other._config, hermes_home=other._hermes_home, create=False
        )
        if path_mode == "payload_symlink":
            storage = get_large_output_storage_dir(
                engine._config, hermes_home=engine._hermes_home, create=True
            )
            (storage / ref).symlink_to(other_storage / ref)
        else:
            engine._config.large_output_externalization_path = str(other_storage)
        # A matching session id in another profile must not grant access.
        assert _expand(other, externalized_ref=ref, session_id="session-a")["content"]
        assert _expand(engine, externalized_ref=ref, session_id="session-a") == {
            "error": f"Externalized payload {ref} not found in session session-a"
        }
    finally:
        other.shutdown()
        engine.shutdown()


@pytest.mark.parametrize("path_mode", [
    "configured_directory", "directory_symlink", "symlinked_profiles_dir",
    "symlinked_named_profile", "payload_target_symlink", "case_variant", "root_default",
])
def test_explicit_session_nested_profile_payload_containment(tmp_path, path_mode):
    if path_mode == "case_variant":
        (tmp_path / "case-probe").write_text("probe", encoding="utf-8")
        if not (tmp_path / "CASE-PROBE").exists():
            pytest.skip("requires a case-insensitive filesystem")
    engine = _rotation_engine(tmp_path, hermes_home=tmp_path)
    profile_parent = {"symlinked_profiles_dir": "alternate", "symlinked_named_profile": "elsewhere"}
    other_home = tmp_path / profile_parent.get(path_mode, "profiles") / "work"
    other_home.mkdir(parents=True)
    if path_mode == "symlinked_profiles_dir":
        (tmp_path / "profiles").symlink_to(other_home.parent, target_is_directory=True)
    elif path_mode == "symlinked_named_profile":
        (tmp_path / "profiles").mkdir()
        (tmp_path / "profiles" / "work").symlink_to(other_home, target_is_directory=True)
    other = _rotation_engine(other_home, hermes_home=other_home)
    try:
        _restart(other, session_id="work-session")
        ref = _externalize(other, "work-session", "RESULT from nested profile " * 50)
        other_storage = get_large_output_storage_dir(
            other._config, hermes_home=other._hermes_home, create=False
        )
        assert _expand(other, externalized_ref=ref, session_id="work-session")["content"]
        if path_mode == "root_default":
            content = "RESULT from root profile " * 50
            ref = _externalize(engine, "session-a", content)
            _restart(engine)
            assert _expand(engine, externalized_ref=ref, session_id="session-a")["content"] == content
            return
        if path_mode == "payload_target_symlink":
            engine._config.large_output_externalization_path = str(tmp_path)
            (tmp_path / ref).symlink_to(other_storage / ref)
        elif path_mode == "case_variant":
            engine._config.large_output_externalization_path = str(
                tmp_path / "Profiles" / other_storage.relative_to(tmp_path / "profiles")
            )
        elif path_mode == "directory_symlink":
            storage = get_large_output_storage_dir(
                engine._config, hermes_home=engine._hermes_home, create=False
            )
            storage.symlink_to(other_storage, target_is_directory=True)
        else:
            engine._config.large_output_externalization_path = str(other_storage.resolve())
        assert _expand(engine, externalized_ref=ref, session_id="work-session") == {
            "error": f"Externalized payload {ref} not found in session work-session"
        }
    finally:
        other.shutdown()
        engine.shutdown()


def test_explicit_session_requires_payload_ownership(tmp_path):
    engine = _rotation_engine(tmp_path)
    try:
        ref = _externalize(engine, "", "RESULT without session ownership " * 50)
        assert _expand(engine, externalized_ref=ref, session_id="session-a") == {
            "error": f"Externalized payload {ref} not found in session session-a"
        }
    finally:
        engine.shutdown()


def test_explicit_session_rejects_path_outside_payload_directory(tmp_path):
    engine = _rotation_engine(tmp_path)
    try:
        ref = _externalize(engine, "session-a", "RESULT in session a " * 50)
        storage = get_large_output_storage_dir(
            engine._config, hermes_home=engine._hermes_home, create=False
        )
        outside = storage.parent / "outside.json"
        outside.write_text((storage / ref).read_text(encoding="utf-8"), encoding="utf-8")
        linked_ref = "linked.json"
        (storage / linked_ref).symlink_to(outside)
        assert _expand(engine, externalized_ref=linked_ref, session_id="session-a") == {
            "error": f"Externalized payload {linked_ref} not found in session session-a"
        }
    finally:
        engine.shutdown()


def test_explicit_session_follows_rotation_lineage(tmp_path):
    engine = _rotation_engine(tmp_path)
    content = "RESULT in session a " * 50
    try:
        ref = _externalize(engine, "session-a", content)
        _rotate(engine, "session-a", "session-b")
        _restart(engine, "session-c")
        result = _expand(engine, externalized_ref=ref, session_id="session-b")
        assert result["content"] == content
        assert result["session_id"] == "session-a"
        result = _expand(engine, externalized_ref=ref, session_id="session-c")
        assert "not found in session session-c" in result["error"]
        assert "content" not in result
    finally:
        engine.shutdown()


def test_store_id_still_rejects_session_id(tmp_path):
    engine = _rotation_engine(tmp_path)
    try:
        store_id = engine._store.append("session-a", {"role": "user", "content": "stored value"})
        result = _expand(engine, store_id=store_id, session_id="session-a")
        assert "session_id is only valid with" in result["error"]
    finally:
        engine.shutdown()


def test_cross_session_store_row_hint_round_trips(tmp_path):
    engine = _rotation_engine(tmp_path)
    content = "RESULT in session a " * 50
    try:
        ref = _externalize(engine, "session-a", content)
        placeholder = (
            "[Externalized tool output: tool_call_id=call-session-a; "
            f"chars={len(content)}; bytes={len(content.encode())}; ref={ref}]"
        )
        store_id = engine._store.append(
            "session-a", {"role": "tool", "content": placeholder, "tool_call_id": "call-session-a"}
        )
        _restart(engine)
        result = _expand(engine, store_id=store_id)
        assert result["externalized_ref"] == ref
        assert "externalized" not in result
        assert "externalized_payloads" not in result
        assert "session-scoped" in result["externalized_note"]
        assert "cannot be expanded" not in result["externalized_note"]
        assert result["externalized_expand_hint"] == (
            f"lcm_expand(externalized_ref='{ref}', session_id='session-a')"
        )
        expanded = _expand(engine, externalized_ref=ref, session_id="session-a")
        assert expanded["content"] == content
    finally:
        engine.shutdown()


def test_legacy_store_row_without_session_has_no_dangling_hint(tmp_path):
    engine = _rotation_engine(tmp_path)
    try:
        content = "RESULT from legacy row " * 50
        ref = _externalize(engine, "", content)
        placeholder = (
            "[Externalized tool output: tool_call_id=call-legacy; "
            f"chars={len(content)}; bytes={len(content.encode())}; ref={ref}]"
        )
        store_id = engine._store.append(
            "", {"role": "tool", "content": placeholder, "tool_call_id": "call-legacy"}
        )
        result = _expand(engine, store_id=store_id)
        assert result["session_id"] == ""
        assert result["externalized_ref"] == ref
        assert "externalized_expand_hint" not in result
        assert result["externalized_note"] == (
            "Externalized payload metadata is session-scoped; "
            "cross-session ref is surfaced for traceability only and cannot be expanded in this version."
        )
    finally:
        engine.shutdown()


def test_no_session_id_responses_unchanged(tmp_path):
    engine = _rotation_engine(tmp_path)
    try:
        ref = _externalize(engine, "session-a", "RESULT in session a " * 50)
        result = _expand(engine, externalized_ref=ref, max_tokens=50)
        assert set(result) == {
            "externalized_ref",
            "source_type",
            "kind",
            "tool_call_id",
            "role",
            "session_id",
            "field_path",
            "content_chars",
            "content_bytes",
            "content",
            "content_offset",
            "content_returned_chars",
            "content_truncated",
            "next_content_offset",
            "has_more",
        }
        assert "externalized_expand_hint" not in result
        described = json.loads(engine.handle_tool_call(
            "lcm_describe", {"externalized_ref": ref, "session_id": "session-a"}
        ))
        assert "cannot be combined with session_id" in described["error"]
    finally:
        engine.shutdown()
