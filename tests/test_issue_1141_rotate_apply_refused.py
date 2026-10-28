"""#1141: Hermes owns transcript replay, so rotate apply must not mutate."""

import pytest

from hermes_lcm.command import _rotate_apply_unchecked, handle_lcm_command
from tests.test_lcm_rotate import _build_engine, _seed_messages


def test_rotate_apply_refused_without_backup_or_lifecycle_change(tmp_path):
    engine = _build_engine(tmp_path, fresh_tail_count=3)
    try:
        _seed_messages(engine, count=10)
        before = engine._lifecycle.get_by_conversation(engine._conversation_id)
        frontier = before.current_frontier_store_id
        last_rotate_at = engine.get_status()["last_rotate_at"]
        backup_path = engine.rotate_backup_path()
        assert not backup_path.exists()

        result = handle_lcm_command("rotate apply", engine)

        assert "status: refused" in result
        assert "reason: host_owns_transcript" in result
        assert "/compress here N" in result
        assert not backup_path.exists()
        after = engine._lifecycle.get_by_conversation(engine._conversation_id)
        assert after.current_frontier_store_id == frontier
        assert engine.get_status()["last_rotate_at"] == last_rotate_at
    finally:
        engine.shutdown()


@pytest.mark.parametrize("command", ["rotate apply", "rotate"])
def test_rotate_names_new_for_already_rotated_session(tmp_path, command):
    engine = _build_engine(tmp_path, fresh_tail_count=3)
    try:
        _seed_messages(engine, count=10)

        result = handle_lcm_command(command, engine)

        assert "/new" in result
        assert "#1184" in result
    finally:
        engine.shutdown()


def test_rotate_noop_preview_names_new_for_already_rotated_session(tmp_path):
    engine = _build_engine(tmp_path, fresh_tail_count=3)
    try:
        _seed_messages(engine, count=10)
        _rotate_apply_unchecked(engine)

        result = handle_lcm_command("rotate", engine)

        assert "status: noop" in result
        assert "reason: frontier_already_ahead" in result
        assert "/new" in result
        assert "#1184" in result
    finally:
        engine.shutdown()


def test_rotate_preview_keeps_planned_frontier_and_names_compress(tmp_path):
    engine = _build_engine(tmp_path, fresh_tail_count=3)
    try:
        _seed_messages(engine, count=10)

        result = handle_lcm_command("rotate", engine)

        assert "status: preview" in result
        assert "current_frontier_store_id: 0" in result
        assert "new_frontier_store_id: 7" in result
        assert "/compress here N" in result
        assert "rotate apply is refused, #1141" in result
    finally:
        engine.shutdown()
