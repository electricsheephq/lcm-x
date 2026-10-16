"""LCM engine integration tests: auxiliary handoff."""

import sqlite3
import threading



from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine

from .engine_helpers import (
    TestSessionRollover as _SessionRolloverHelpers,
)


class TestSessionRollover:
    HostAgentFrame = _SessionRolloverHelpers.__dict__["HostAgentFrame"]
    _start_host_child = _SessionRolloverHelpers.__dict__["_start_host_child"]

    def test_stale_auxiliary_end_does_not_reset_live_native_fallback(self, tmp_path):
        class FakeContextCompressor:
            def __init__(self):
                self.ended = []
                self.reset = False

            def on_session_end(self, session_id, messages):
                self.ended.append((session_id, messages))

            def on_session_reset(self):
                self.reset = True

        hermes_home = tmp_path / "hermes-home"
        hermes_home.mkdir()
        config = LCMConfig(database_path=str(tmp_path / "lcm_stale_aux_end_fallback.db"))
        engine = LCMEngine(config=config, hermes_home=str(hermes_home))
        engine.on_session_start(
            "foreground-session",
            hermes_home=str(hermes_home),
            platform="telegram",
            context_length=1_000,
        )
        engine.threshold_tokens = 20
        stale_child = self._start_host_child(
            engine,
            hermes_home,
            "background-review-session",
            "foreground-session",
        )
        live_child = self._start_host_child(
            engine,
            hermes_home,
            "background-review-session",
            "foreground-session",
        )
        live_child.update_from_response(engine, {"prompt_tokens": 25})
        compressor = FakeContextCompressor()
        engine._host_fallback_compressor = compressor
        engine._host_fallback_session_id = "background-review-session"

        stale_child.on_session_end(engine, [{"role": "user", "content": "stale final"}])

        assert compressor.ended == []
        assert compressor.reset is False
        assert engine._host_fallback_compressor is compressor
        assert engine._host_fallback_session_id == "background-review-session"
        assert engine._thread_context_has_auxiliary_session("background-review-session")
        assert live_child.should_compress(engine)

    def test_auxiliary_generation_replacement_resets_native_fallback(self, tmp_path):
        class FakeContextCompressor:
            def __init__(self):
                self.ended = []
                self.reset = False

            def on_session_end(self, session_id, messages):
                self.ended.append((session_id, messages))

            def on_session_reset(self):
                self.reset = True

        hermes_home = tmp_path / "hermes-home"
        hermes_home.mkdir()
        config = LCMConfig(database_path=str(tmp_path / "lcm_aux_replacement_fallback_reset.db"))
        engine = LCMEngine(config=config, hermes_home=str(hermes_home))
        engine.on_session_start(
            "foreground-session",
            hermes_home=str(hermes_home),
            platform="telegram",
            context_length=1_000,
        )
        old_child = self._start_host_child(
            engine,
            hermes_home,
            "background-review-session",
            "foreground-session",
        )
        old_child.update_from_response(engine, {"prompt_tokens": 25})
        compressor = FakeContextCompressor()
        engine._host_fallback_compressor = compressor
        engine._host_fallback_session_id = "background-review-session"

        live_child = self._start_host_child(
            engine,
            hermes_home,
            "background-review-session",
            "foreground-session",
        )

        assert compressor.ended == [("background-review-session", [])]
        assert compressor.reset is True
        assert engine._host_fallback_compressor is None
        assert engine._host_fallback_session_id == ""
        live_child.update_from_response(engine, {"prompt_tokens": 30})
        assert engine._auxiliary_last_prompt_tokens == {"background-review-session": 30}

    def test_auxiliary_generation_adoption_resets_native_fallback(self, tmp_path):
        class FakeContextCompressor:
            def __init__(self):
                self.ended = []
                self.reset = False

            def on_session_end(self, session_id, messages):
                self.ended.append((session_id, messages))

            def on_session_reset(self):
                self.reset = True

        hermes_home = tmp_path / "hermes-home"
        hermes_home.mkdir()
        config = LCMConfig(database_path=str(tmp_path / "lcm_aux_adoption_fallback_reset.db"))
        engine = LCMEngine(config=config, hermes_home=str(hermes_home))
        engine.on_session_start(
            "foreground-session",
            hermes_home=str(hermes_home),
            platform="telegram",
            context_length=1_000,
        )
        child = self.HostAgentFrame(
            "background-review-session",
            "foreground-session",
            hermes_home,
        )
        engine._mark_thread_context_stateless("background-review-session")
        compressor = FakeContextCompressor()
        engine._host_fallback_compressor = compressor
        engine._host_fallback_session_id = "background-review-session"

        child.update_from_response(engine, {"prompt_tokens": 30})

        assert compressor.ended == [("background-review-session", [])]
        assert compressor.reset is True
        assert engine._host_fallback_compressor is None
        assert engine._host_fallback_session_id == ""
        assert engine._auxiliary_last_prompt_tokens == {"background-review-session": 30}
        assert "background-review-session" in engine._auxiliary_session_generations

    def test_auxiliary_handoff_generation_replacement_resets_native_fallback(self, tmp_path):
        class FakeContextCompressor:
            def __init__(self):
                self.ended = []
                self.reset = False

            def on_session_end(self, session_id, messages):
                self.ended.append((session_id, messages))

            def on_session_reset(self):
                self.reset = True

        hermes_home = tmp_path / "hermes-home"
        hermes_home.mkdir()
        config = LCMConfig(database_path=str(tmp_path / "lcm_aux_handoff_replacement_fallback_reset.db"))
        engine = LCMEngine(config=config, hermes_home=str(hermes_home))
        engine.on_session_start(
            "foreground-session",
            hermes_home=str(hermes_home),
            platform="telegram",
            context_length=1_000,
        )
        self._start_host_child(engine, hermes_home, "parent-session", "foreground-session")
        first_continuation = self.HostAgentFrame(
            "reused-continuation",
            "parent-session",
            hermes_home,
        )
        first_continuation.on_compression_boundary_from(
            engine,
            old_session_id="parent-session",
            context_length=1_000,
        )
        first_generation = engine._auxiliary_session_generations["reused-continuation"]
        compressor = FakeContextCompressor()
        engine._host_fallback_compressor = compressor
        engine._host_fallback_session_id = "reused-continuation"

        second_continuation = self.HostAgentFrame(
            "reused-continuation",
            "parent-session",
            hermes_home,
        )
        second_continuation.on_compression_boundary_from(
            engine,
            old_session_id="parent-session",
            context_length=1_000,
        )

        assert engine._auxiliary_session_generations["reused-continuation"] != first_generation
        assert compressor.ended == [("reused-continuation", [])]
        assert compressor.reset is True
        assert engine._host_fallback_compressor is None
        assert engine._host_fallback_session_id == ""

    def test_stale_auxiliary_boundary_does_not_reset_live_native_fallback(self, tmp_path):
        class FakeContextCompressor:
            def __init__(self):
                self.ended = []
                self.reset = False

            def on_session_end(self, session_id, messages):
                self.ended.append((session_id, messages))

            def on_session_reset(self):
                self.reset = True

        hermes_home = tmp_path / "hermes-home"
        hermes_home.mkdir()
        config = LCMConfig(database_path=str(tmp_path / "lcm_stale_aux_boundary_fallback.db"))
        engine = LCMEngine(config=config, hermes_home=str(hermes_home))
        engine.on_session_start(
            "foreground-session",
            hermes_home=str(hermes_home),
            platform="telegram",
            context_length=1_000,
        )
        stale_child = self._start_host_child(
            engine,
            hermes_home,
            "background-review-session",
            "foreground-session",
        )
        live_child = self._start_host_child(
            engine,
            hermes_home,
            "background-review-session",
            "foreground-session",
        )
        live_child.update_from_response(engine, {"prompt_tokens": 30})
        compressor = FakeContextCompressor()
        engine._host_fallback_compressor = compressor
        engine._host_fallback_session_id = "background-review-session"

        stale_child.on_compression_boundary(engine, "stale-continuation", context_length=1_000)

        assert compressor.ended == []
        assert compressor.reset is False
        assert engine._host_fallback_compressor is compressor
        assert engine._host_fallback_session_id == "background-review-session"
        assert engine._thread_context_has_auxiliary_session("background-review-session")
        assert not engine._thread_context_has_auxiliary_session("stale-continuation")
        assert engine._auxiliary_last_prompt_tokens == {"background-review-session": 30}
        engine.threshold_tokens = 20
        assert live_child.should_compress(engine)

    def test_direct_host_end_clears_guarded_replacement_with_usage(self, tmp_path):
        hermes_home = tmp_path / "hermes-home"
        hermes_home.mkdir()
        config = LCMConfig(database_path=str(tmp_path / "lcm_guarded_replacement_direct_end.db"))
        engine = LCMEngine(config=config, hermes_home=str(hermes_home))
        engine.on_session_start(
            "foreground-session",
            hermes_home=str(hermes_home),
            platform="telegram",
            context_length=1_000,
        )
        old_child = self._start_host_child(
            engine,
            hermes_home,
            "background-review-session",
            "foreground-session",
        )
        old_child.update_from_response(engine, {"prompt_tokens": 25})
        replacement_child = self._start_host_child(
            engine,
            hermes_home,
            "background-review-session",
            "foreground-session",
        )
        replacement_child.update_from_response(engine, {"prompt_tokens": 30})
        assert "background-review-session" in engine._auxiliary_direct_end_guard_session_ids
        assert engine._auxiliary_last_prompt_tokens == {"background-review-session": 30}

        engine.on_session_end("background-review-session", [])

        assert not engine._thread_context_has_auxiliary_session("background-review-session")
        assert engine._auxiliary_last_prompt_tokens == {}
        assert "background-review-session" not in engine._auxiliary_direct_end_guard_session_ids
        assert "background-review-session" not in engine._auxiliary_session_generations

    def test_generationless_duplicate_start_preserves_live_auxiliary_generation(self, tmp_path):
        hermes_home = tmp_path / "hermes-home"
        hermes_home.mkdir()

        config = LCMConfig(database_path=str(tmp_path / "lcm_aux_duplicate_start_generation.db"))
        engine = LCMEngine(config=config, hermes_home=str(hermes_home))
        engine.on_session_start(
            "foreground-session",
            hermes_home=str(hermes_home),
            platform="telegram",
            context_length=1_000,
        )
        engine.threshold_tokens = 20

        child = self._start_host_child(
            engine,
            hermes_home,
            "background-review-session",
            "foreground-session",
        )
        previous_generation = engine._auxiliary_session_generations["background-review-session"]

        child.on_session_start(
            engine,
            parent_session_id="foreground-session",
            context_length=1_000,
        )

        assert engine._auxiliary_session_generations == {
            "background-review-session": previous_generation
        }
        assert engine._auxiliary_retired_session_generations == {}
        child.update_from_response(engine, {"prompt_tokens": 25})
        assert engine._auxiliary_last_prompt_tokens == {"background-review-session": 25}
        assert child.should_compress(engine)

    def test_auxiliary_handoff_reused_generationless_continuation_rejects_stale_parent(self, tmp_path):
        hermes_home = tmp_path / "hermes-home"
        hermes_home.mkdir()

        config = LCMConfig(database_path=str(tmp_path / "lcm_aux_handoff_reused_generationless.db"))
        engine = LCMEngine(config=config, hermes_home=str(hermes_home))
        engine.on_session_start(
            "foreground-session",
            hermes_home=str(hermes_home),
            platform="telegram",
            context_length=1_000,
        )
        engine.threshold_tokens = 20

        parent_one = self._start_host_child(
            engine,
            hermes_home,
            "parent-one",
            "foreground-session",
        )
        parent_one.on_compression_boundary(
            engine,
            "shared-continuation",
            context_length=1_000,
        )
        stale_continuation = self.HostAgentFrame(
            "shared-continuation",
            "parent-one",
            hermes_home,
        )

        parent_two = self._start_host_child(
            engine,
            hermes_home,
            "parent-two",
            "foreground-session",
        )
        parent_two.on_compression_boundary(
            engine,
            "shared-continuation",
            context_length=1_000,
        )
        real_continuation = self.HostAgentFrame(
            "shared-continuation",
            "parent-two",
            hermes_home,
        )

        assert engine._auxiliary_handoff_parent_session_ids == {
            "shared-continuation": "parent-two"
        }
        stale_continuation.update_from_response(engine, {"prompt_tokens": 25})
        assert engine._auxiliary_last_prompt_tokens == {}
        assert engine._auxiliary_session_generations == {}

        real_continuation.update_from_response(engine, {"prompt_tokens": 30})
        assert engine._auxiliary_last_prompt_tokens == {"shared-continuation": 30}
        assert real_continuation.should_compress(engine)
        assert not stale_continuation.should_compress(engine)

    def test_stale_generationless_boundary_preserves_live_continuation_usage(self, tmp_path):
        hermes_home = tmp_path / "hermes-home"
        hermes_home.mkdir()

        config = LCMConfig(database_path=str(tmp_path / "lcm_aux_stale_boundary_live_usage.db"))
        engine = LCMEngine(config=config, hermes_home=str(hermes_home))
        engine.on_session_start(
            "foreground-session",
            hermes_home=str(hermes_home),
            platform="telegram",
            context_length=1_000,
        )
        engine.threshold_tokens = 20

        parent_one = self._start_host_child(engine, hermes_home, "parent-one", "foreground-session")
        parent_one.on_compression_boundary(engine, "shared-continuation", context_length=1_000)
        parent_two = self._start_host_child(engine, hermes_home, "parent-two", "foreground-session")
        parent_two.on_compression_boundary(engine, "shared-continuation", context_length=1_000)
        real_continuation = self.HostAgentFrame(
            "shared-continuation",
            "parent-two",
            hermes_home,
        )
        real_continuation.update_from_response(engine, {"prompt_tokens": 30})
        assert real_continuation.should_compress(engine)

        parent_one.on_compression_boundary(engine, "shared-continuation", context_length=1_000)

        assert engine._auxiliary_last_prompt_tokens == {"shared-continuation": 30}
        assert real_continuation.should_compress(engine)

    def test_no_frame_stale_boundary_does_not_replace_live_auxiliary(self, tmp_path):
        class FakeContextCompressor:
            def __init__(self):
                self.ended = []
                self.reset = False

            def on_session_end(self, session_id, messages):
                self.ended.append((session_id, messages))

            def on_session_reset(self):
                self.reset = True

        hermes_home = tmp_path / "hermes-home"
        hermes_home.mkdir()

        config = LCMConfig(database_path=str(tmp_path / "lcm_no_frame_stale_boundary_live_aux.db"))
        engine = LCMEngine(config=config, hermes_home=str(hermes_home))
        engine.on_session_start(
            "foreground-session",
            hermes_home=str(hermes_home),
            platform="telegram",
            context_length=1_000,
        )
        engine.threshold_tokens = 20
        stale_child = self._start_host_child(engine, hermes_home, "reused-child", "foreground-session")
        stale_child.on_session_end(engine, [])
        live_child = self._start_host_child(engine, hermes_home, "reused-child", "foreground-session")
        live_generation = engine._auxiliary_session_generations["reused-child"]
        live_child.update_from_response(engine, {"prompt_tokens": 30})
        compressor = FakeContextCompressor()
        engine._host_fallback_compressor = compressor
        engine._host_fallback_session_id = "reused-child"
        assert live_child.should_compress(engine)

        engine.on_session_start(
            "stale-continuation",
            hermes_home=str(hermes_home),
            boundary_reason="compression",
            old_session_id="reused-child",
            platform="telegram",
            context_length=1_000,
        )

        assert engine._thread_context_has_auxiliary_session("reused-child")
        assert engine._thread_context_has_auxiliary_session("stale-continuation")
        assert engine._auxiliary_session_generations["reused-child"] == live_generation
        assert engine._auxiliary_last_prompt_tokens == {"reused-child": 30}
        assert engine._host_fallback_compressor is compressor
        assert engine._host_fallback_session_id == "reused-child"
        assert compressor.ended == []
        assert compressor.reset is False
        assert live_child.should_compress(engine)
        assert engine._store.get_session_count("reused-child") == 0
        assert engine._store.get_session_count("stale-continuation") == 0

    def test_auxiliary_handoff_to_foreground_reused_id_stays_active(self, tmp_path):
        hermes_home = tmp_path / "hermes-home"
        hermes_home.mkdir()

        config = LCMConfig(database_path=str(tmp_path / "lcm_aux_handoff_foreground_reused_id.db"))
        engine = LCMEngine(config=config, hermes_home=str(hermes_home))
        engine.on_session_start(
            "foreground-session",
            hermes_home=str(hermes_home),
            platform="telegram",
            context_length=1_000,
        )
        engine.threshold_tokens = 20
        old_aux = self._start_host_child(
            engine,
            hermes_home,
            "reused-continuation",
            "foreground-session",
        )
        old_aux.update_from_response(engine, {"prompt_tokens": 25})
        engine.on_session_start(
            "reused-continuation",
            hermes_home=str(hermes_home),
            platform="telegram",
            context_length=1_000,
        )
        assert "reused-continuation" in engine._auxiliary_foreground_reused_session_ids

        engine.on_session_start(
            "next-foreground-session",
            hermes_home=str(hermes_home),
            platform="telegram",
            context_length=1_000,
        )
        self._start_host_child(engine, hermes_home, "parent-session", "next-foreground-session")
        continuation = self.HostAgentFrame("reused-continuation", "parent-session", hermes_home)
        continuation.on_compression_boundary_from(engine, old_session_id="parent-session")

        assert engine._thread_context_has_auxiliary_session("reused-continuation")
        assert "reused-continuation" in engine._auxiliary_foreground_reused_session_ids
        assert engine._auxiliary_session_generations["reused-continuation"]
        continuation.update_from_response(engine, {"prompt_tokens": 25})
        assert engine._auxiliary_last_prompt_tokens == {"reused-continuation": 25}
        engine.threshold_tokens = 20
        engine._last_boundary_skip_time = 0
        assert continuation.should_compress(engine)
        assert engine._store.get_session_count("reused-continuation") == 0
        assert engine._store.get_session_count("parent-session") == 0

    def test_foreground_reused_handoff_replacement_clears_stale_usage(self, tmp_path):
        hermes_home = tmp_path / "hermes-home"
        hermes_home.mkdir()

        config = LCMConfig(database_path=str(tmp_path / "lcm_foreground_reused_handoff_replacement.db"))
        engine = LCMEngine(config=config, hermes_home=str(hermes_home))
        engine.on_session_start(
            "foreground-session",
            hermes_home=str(hermes_home),
            platform="telegram",
            context_length=1_000,
        )
        engine.threshold_tokens = 20
        old_aux = self._start_host_child(
            engine,
            hermes_home,
            "reused-continuation",
            "foreground-session",
        )
        old_aux.update_from_response(engine, {"prompt_tokens": 25})
        engine.on_session_start(
            "reused-continuation",
            hermes_home=str(hermes_home),
            platform="telegram",
            context_length=1_000,
        )
        assert "reused-continuation" in engine._auxiliary_foreground_reused_session_ids

        engine.on_session_start(
            "next-foreground-session",
            hermes_home=str(hermes_home),
            platform="telegram",
            context_length=1_000,
        )
        self._start_host_child(engine, hermes_home, "parent-a", "next-foreground-session")
        first_continuation = self.HostAgentFrame("reused-continuation", "parent-a", hermes_home)
        first_continuation.on_compression_boundary_from(engine, old_session_id="parent-a")
        first_continuation.update_from_response(engine, {"prompt_tokens": 30})
        first_generation = engine._auxiliary_session_generations["reused-continuation"]
        assert engine._auxiliary_last_prompt_tokens == {"reused-continuation": 30}

        self._start_host_child(engine, hermes_home, "parent-b", "next-foreground-session")
        second_continuation = self.HostAgentFrame("reused-continuation", "parent-b", hermes_home)
        second_continuation.on_compression_boundary_from(engine, old_session_id="parent-b")

        assert engine._auxiliary_session_generations["reused-continuation"] != first_generation
        assert "reused-continuation" in engine._auxiliary_foreground_reused_session_ids
        assert engine._auxiliary_last_prompt_tokens == {}
        assert not second_continuation.should_compress(engine)
        assert not first_continuation.should_compress(engine)
        second_continuation.update_from_response(engine, {"prompt_tokens": 30})
        assert engine._auxiliary_last_prompt_tokens == {"reused-continuation": 30}

    def test_parent_handoff_to_foreground_reused_id_retires_displaced_generation(self, tmp_path):
        hermes_home = tmp_path / "hermes-home"
        hermes_home.mkdir()

        config = LCMConfig(database_path=str(tmp_path / "lcm_parent_handoff_retires_reused_generation.db"))
        engine = LCMEngine(config=config, hermes_home=str(hermes_home))
        engine.on_session_start(
            "foreground-session",
            hermes_home=str(hermes_home),
            platform="telegram",
            context_length=1_000,
        )
        engine.threshold_tokens = 20
        old_aux = self._start_host_child(
            engine,
            hermes_home,
            "reused-continuation",
            "foreground-session",
        )
        old_aux.update_from_response(engine, {"prompt_tokens": 25})
        engine.on_session_start(
            "reused-continuation",
            hermes_home=str(hermes_home),
            platform="telegram",
            context_length=1_000,
        )
        engine.on_session_start(
            "next-foreground-session",
            hermes_home=str(hermes_home),
            platform="telegram",
            context_length=1_000,
        )
        self._start_host_child(engine, hermes_home, "parent-a", "next-foreground-session")
        first_continuation = self.HostAgentFrame("reused-continuation", "parent-a", hermes_home)
        first_continuation.on_compression_boundary_from(engine, old_session_id="parent-a")
        first_continuation.update_from_response(engine, {"prompt_tokens": 30})
        first_generation = engine._auxiliary_session_generations["reused-continuation"]

        parent_b = self._start_host_child(engine, hermes_home, "parent-b", "next-foreground-session")
        parent_b.on_compression_boundary(engine, "reused-continuation", context_length=1_000)

        assert engine._auxiliary_generation_is_retired("reused-continuation", first_generation)
        assert engine._auxiliary_last_prompt_tokens == {}
        first_continuation.update_from_response(engine, {"prompt_tokens": 99})
        assert engine._auxiliary_last_prompt_tokens == {}
        assert not first_continuation.should_compress(engine)
        assert engine._store.get_session_count("reused-continuation") == 0

    def test_stale_parent_handoff_does_not_clobber_live_reused_continuation(self, tmp_path):
        hermes_home = tmp_path / "hermes-home"
        hermes_home.mkdir()

        config = LCMConfig(database_path=str(tmp_path / "lcm_stale_parent_handoff_live_reused.db"))
        engine = LCMEngine(config=config, hermes_home=str(hermes_home))
        engine.on_session_start(
            "foreground-session",
            hermes_home=str(hermes_home),
            platform="telegram",
            context_length=1_000,
        )
        engine.threshold_tokens = 20
        old_aux = self._start_host_child(
            engine,
            hermes_home,
            "reused-continuation",
            "foreground-session",
        )
        old_aux.update_from_response(engine, {"prompt_tokens": 25})
        engine.on_session_start(
            "reused-continuation",
            hermes_home=str(hermes_home),
            platform="telegram",
            context_length=1_000,
        )
        engine.on_session_start(
            "next-foreground-session",
            hermes_home=str(hermes_home),
            platform="telegram",
            context_length=1_000,
        )
        self._start_host_child(engine, hermes_home, "parent-a", "next-foreground-session")
        first_continuation = self.HostAgentFrame("reused-continuation", "parent-a", hermes_home)
        first_continuation.on_compression_boundary_from(engine, old_session_id="parent-a")
        first_continuation.update_from_response(engine, {"prompt_tokens": 30})
        self._start_host_child(engine, hermes_home, "parent-b", "next-foreground-session")
        second_continuation = self.HostAgentFrame("reused-continuation", "parent-b", hermes_home)
        second_continuation.on_compression_boundary_from(engine, old_session_id="parent-b")
        second_continuation.update_from_response(engine, {"prompt_tokens": 40})
        live_generation = engine._auxiliary_session_generations["reused-continuation"]
        retired_parent_a_generations = engine._auxiliary_retired_session_generations["parent-a"]
        assert retired_parent_a_generations
        assert all(
            engine._auxiliary_generation_is_retired("parent-a", generation)
            for generation in retired_parent_a_generations
        )

        engine.on_session_start(
            "reused-continuation",
            hermes_home=str(hermes_home),
            boundary_reason="compression",
            old_session_id="parent-a",
            platform="telegram",
            context_length=1_000,
        )

        assert engine._auxiliary_session_generations["reused-continuation"] == live_generation
        assert engine._auxiliary_last_prompt_tokens == {"reused-continuation": 40}
        second_continuation.update_from_response(engine, {"prompt_tokens": 45})
        assert engine._auxiliary_last_prompt_tokens == {"reused-continuation": 45}
        assert engine._store.get_session_count("reused-continuation") == 0

    def test_generationless_parent_handoff_to_foreground_reused_id_stays_active(self, tmp_path):
        hermes_home = tmp_path / "hermes-home"
        hermes_home.mkdir()

        config = LCMConfig(database_path=str(tmp_path / "lcm_aux_generationless_handoff_foreground_reused_id.db"))
        engine = LCMEngine(config=config, hermes_home=str(hermes_home))
        engine.on_session_start(
            "foreground-session",
            hermes_home=str(hermes_home),
            platform="telegram",
            context_length=1_000,
        )
        old_aux = self._start_host_child(
            engine,
            hermes_home,
            "reused-continuation",
            "foreground-session",
        )
        old_aux.update_from_response(engine, {"prompt_tokens": 25})
        engine.on_session_start(
            "reused-continuation",
            hermes_home=str(hermes_home),
            platform="telegram",
            context_length=1_000,
        )
        assert "reused-continuation" in engine._auxiliary_foreground_reused_session_ids

        engine.on_session_start(
            "next-foreground-session",
            hermes_home=str(hermes_home),
            platform="telegram",
            context_length=1_000,
        )
        parent = self._start_host_child(engine, hermes_home, "parent-session", "next-foreground-session")
        parent.on_compression_boundary(engine, "reused-continuation", context_length=1_000)

        assert engine._thread_context_has_auxiliary_session("reused-continuation")
        assert "reused-continuation" in engine._auxiliary_foreground_reused_session_ids
        assert "reused-continuation" not in engine._auxiliary_session_generations
        continuation = self.HostAgentFrame("reused-continuation", "parent-session", hermes_home)
        continuation.update_from_response(engine, {"prompt_tokens": 25})
        assert engine._auxiliary_session_generations["reused-continuation"]
        assert engine._auxiliary_last_prompt_tokens == {"reused-continuation": 25}
        engine.threshold_tokens = 20
        engine._last_boundary_skip_time = 0
        assert continuation.should_compress(engine)

    def test_stale_retired_auxiliary_boundary_does_not_clobber_live_generation(self, tmp_path):
        hermes_home = tmp_path / "hermes-home"
        hermes_home.mkdir()

        config = LCMConfig(database_path=str(tmp_path / "lcm_stale_retired_aux_boundary.db"))
        engine = LCMEngine(config=config, hermes_home=str(hermes_home))
        engine.on_session_start(
            "foreground-session",
            hermes_home=str(hermes_home),
            platform="telegram",
            context_length=1_000,
        )
        stale_child = self._start_host_child(
            engine,
            hermes_home,
            "reused-child",
            "foreground-session",
        )
        stale_child.update_from_response(engine, {"prompt_tokens": 25})

        live_child = self.HostAgentFrame("reused-child", "foreground-session", hermes_home)
        live_child.on_session_start(engine, context_length=1_000)
        live_child.update_from_response(engine, {"prompt_tokens": 30})
        live_generation = engine._auxiliary_session_generations["reused-child"]
        assert live_generation != 0
        assert engine._auxiliary_last_prompt_tokens == {"reused-child": 30}

        stale_child.on_compression_boundary(engine, "stale-continuation", context_length=1_000)

        assert engine._thread_context_has_auxiliary_session("reused-child")
        assert engine._auxiliary_session_generations["reused-child"] == live_generation
        assert engine._auxiliary_last_prompt_tokens == {"reused-child": 30}
        assert live_child.thread_context_session_id(engine) == "reused-child"

    def test_generationless_stale_auxiliary_boundary_does_not_clobber_live_generation(self, tmp_path):
        hermes_home = tmp_path / "hermes-home"
        hermes_home.mkdir()

        config = LCMConfig(database_path=str(tmp_path / "lcm_generationless_stale_aux_boundary.db"))
        engine = LCMEngine(config=config, hermes_home=str(hermes_home))
        engine.on_session_start(
            "foreground-session",
            hermes_home=str(hermes_home),
            platform="telegram",
            context_length=1_000,
        )
        stale_child = self._start_host_child(
            engine,
            hermes_home,
            "reused-child",
            "foreground-session",
        )
        stale_child.update_from_response(engine, {"prompt_tokens": 25})

        live_child = self.HostAgentFrame("reused-child", "foreground-session", hermes_home)
        live_child.on_session_start(engine, context_length=1_000)
        live_child.update_from_response(engine, {"prompt_tokens": 30})
        live_generation = engine._auxiliary_session_generations["reused-child"]
        assert live_generation != 0
        assert engine._auxiliary_last_prompt_tokens == {"reused-child": 30}

        stale_child.on_compression_boundary(
            engine,
            "stale-continuation",
            context_length=1_000,
        )

        assert engine._thread_context_has_auxiliary_session("reused-child")
        assert not engine._thread_context_has_auxiliary_session("stale-continuation")
        assert engine._auxiliary_session_generations["reused-child"] == live_generation
        assert engine._auxiliary_last_prompt_tokens == {"reused-child": 30}
        engine.threshold_tokens = 20
        assert live_child.should_compress(engine)

    def test_auxiliary_handoff_reused_clean_continuation_rejects_stale_generation(self, tmp_path):
        hermes_home = tmp_path / "hermes-home"
        hermes_home.mkdir()

        config = LCMConfig(database_path=str(tmp_path / "lcm_aux_handoff_reused_clean_continuation.db"))
        engine = LCMEngine(config=config, hermes_home=str(hermes_home))
        engine.on_session_start(
            "foreground-session",
            hermes_home=str(hermes_home),
            platform="telegram",
            context_length=1_000,
        )
        engine.threshold_tokens = 20

        old_child = self._start_host_child(
            engine,
            hermes_home,
            "background-review-session",
            "foreground-session",
        )
        old_child.on_compression_boundary(
            engine,
            "background-review-continuation",
            context_length=1_000,
        )
        stale_continuation = self.HostAgentFrame(
            "background-review-continuation",
            "background-review-session",
            hermes_home,
        )
        stale_continuation.on_session_end(engine, [])
        assert not engine._thread_context_has_auxiliary_session("background-review-continuation")

        old_child = self._start_host_child(
            engine,
            hermes_home,
            "background-review-session",
            "foreground-session",
        )
        old_child.on_compression_boundary(
            engine,
            "background-review-continuation",
            context_length=1_000,
        )
        self.HostAgentFrame(
            "background-review-continuation",
            "background-review-session",
            hermes_home,
        )
        assert "background-review-continuation" not in engine._auxiliary_direct_end_guard_session_ids
        assert engine._auxiliary_handoff_parent_session_ids == {}

        engine.on_session_end("background-review-continuation", [])
        assert not engine._thread_context_has_auxiliary_session("background-review-continuation")
        assert engine._auxiliary_last_prompt_tokens == {}
        assert engine._auxiliary_session_generations == {}
        assert engine._store.get_session_count("background-review-session") == 0
        assert engine._store.get_session_count("background-review-continuation") == 0

    def test_live_guarded_auxiliary_boundary_without_frame_is_accepted(self, tmp_path):
        hermes_home = tmp_path / "hermes-home"
        hermes_home.mkdir()

        config = LCMConfig(database_path=str(tmp_path / "lcm_live_guarded_boundary_no_frame.db"))
        engine = LCMEngine(config=config, hermes_home=str(hermes_home))
        engine.on_session_start(
            "foreground-session",
            hermes_home=str(hermes_home),
            platform="telegram",
            context_length=1_000,
        )
        first_child = self._start_host_child(
            engine,
            hermes_home,
            "background-review-session",
            "foreground-session",
        )
        first_child.update_from_response(engine, {"prompt_tokens": 25})
        live_child = self._start_host_child(
            engine,
            hermes_home,
            "background-review-session",
            "foreground-session",
        )
        live_child.update_from_response(engine, {"prompt_tokens": 30})
        assert "background-review-session" in engine._auxiliary_direct_end_guard_session_ids

        engine.on_session_start(
            "background-review-continuation",
            hermes_home=str(hermes_home),
            boundary_reason="compression",
            old_session_id="background-review-session",
            platform="telegram",
            context_length=1_000,
        )

        assert engine._thread_context_has_auxiliary_session("background-review-continuation")
        assert not engine._thread_context_has_auxiliary_session("background-review-session")
        assert engine._store.get_session_count("background-review-session") == 0
        assert engine._store.get_session_count("background-review-continuation") == 0

    def test_auxiliary_usage_update_racing_session_end_does_not_restore_stale_tokens(
        self,
        tmp_path,
        monkeypatch,
    ):
        hermes_home = tmp_path / "hermes-home"
        hermes_home.mkdir()

        config = LCMConfig(database_path=str(tmp_path / "lcm_aux_usage_race.db"))
        engine = LCMEngine(config=config, hermes_home=str(hermes_home))
        engine.on_session_start(
            "foreground-session",
            hermes_home=str(hermes_home),
            platform="telegram",
            context_length=1_000,
        )
        child = self._start_host_child(
            engine,
            hermes_home,
            "background-review-session",
            "foreground-session",
        )

        reached_active_probe = threading.Event()
        continue_update = threading.Event()
        original_active_auxiliary_session_ids = engine._active_auxiliary_session_ids

        def delayed_active_auxiliary_session_ids():
            if threading.current_thread() is thread:
                reached_active_probe.set()
                assert continue_update.wait(timeout=5)
            return original_active_auxiliary_session_ids()

        monkeypatch.setattr(
            engine,
            "_active_auxiliary_session_ids",
            delayed_active_auxiliary_session_ids,
        )
        errors = []

        def late_update():
            try:
                child.update_from_response(engine, {"prompt_tokens": 25})
            except Exception as exc:  # pragma: no cover - assertion helper
                errors.append(exc)

        thread = threading.Thread(target=late_update)
        thread.start()
        assert reached_active_probe.wait(timeout=5)

        child.on_session_end(engine, [])
        assert not engine._thread_context_has_auxiliary_session("background-review-session")
        continue_update.set()
        thread.join(timeout=5)

        assert not thread.is_alive()
        assert errors == []
        assert engine._auxiliary_last_prompt_tokens == {}
        assert engine._store.get_session_count("foreground-session") == 0
        assert engine._store.get_session_count("background-review-session") == 0

    def test_ended_auxiliary_marker_stays_stateless_until_next_normal_session_start(self, tmp_path):
        hermes_home = tmp_path / "hermes-home"
        hermes_home.mkdir()
        state_db = hermes_home / "state.db"
        conn = sqlite3.connect(state_db)
        conn.executescript(
            """
            CREATE TABLE sessions (
                id TEXT PRIMARY KEY,
                parent_session_id TEXT,
                started_at REAL NOT NULL,
                ended_at REAL
            );
            INSERT INTO sessions(id, parent_session_id, started_at, ended_at)
            VALUES ('foreground-session', NULL, 1.0, NULL);
            INSERT INTO sessions(id, parent_session_id, started_at, ended_at)
            VALUES ('background-review-session', 'foreground-session', 2.0, NULL);
            """
        )
        conn.commit()
        conn.close()

        config = LCMConfig(database_path=str(tmp_path / "lcm_stale_inactive_marker.db"))
        engine = LCMEngine(config=config, hermes_home=str(hermes_home))
        engine.on_session_start(
            "foreground-session",
            hermes_home=str(hermes_home),
            platform="telegram",
            context_length=200000,
        )
        child = self._start_host_child(
            engine,
            hermes_home,
            "background-review-session",
            "foreground-session",
        )
        child.should_compress_preflight(engine, [
            {"role": "user", "content": "child worker must become stateless"},
        ])
        assert child.thread_context_session_id(engine) == "background-review-session"
        assert child.thread_context_stateless(engine)
        assert engine._thread_context_session_id() == ""
        assert not engine._thread_context_stateless()

        errors = []

        def end_auxiliary_child():
            try:
                engine.on_session_end("background-review-session", [])
            except Exception as exc:  # pragma: no cover - assertion helper
                errors.append(exc)

        thread = threading.Thread(target=end_auxiliary_child)
        thread.start()
        thread.join(timeout=5)

        assert not thread.is_alive()
        assert errors == []
        assert not engine._thread_context_has_auxiliary_session("background-review-session")
        assert engine._thread_context_session_id() == ""
        assert not engine._thread_context_stateless()

        late_child_messages = [
            {"role": "user", "content": "late child callback must stay stateless after cross-thread end"},
        ]
        child.should_compress_preflight(engine, late_child_messages)

        assert engine._store.get_session_count("foreground-session") == 0
        assert engine._store.get_session_count("background-review-session") == 0

        engine.should_compress_preflight([
            {"role": "user", "content": "bare foreground callback persists after cross-thread child end"},
        ])
        assert engine._store.get_session_count("foreground-session") == 1

        engine.on_session_start(
            "foreground-followup-session",
            hermes_home=str(hermes_home),
            platform="telegram",
            context_length=200000,
        )
        assert engine._thread_context_session_id() == ""
        assert not engine._thread_context_stateless()

        messages = [{"role": "user", "content": "foreground must persist after normal start"}]
        engine.should_compress_preflight(messages)

        assert engine._store.get_session_count("foreground-followup-session") == 1
        assert engine._store.get_session_count("foreground-session") == 1
        assert engine._store.get_session_count("background-review-session") == 0

    def test_historical_child_session_is_not_treated_as_live_auxiliary(self, tmp_path):
        hermes_home = tmp_path / "hermes-home"
        hermes_home.mkdir()
        state_db = hermes_home / "state.db"
        conn = sqlite3.connect(state_db)
        conn.executescript(
            """
            CREATE TABLE sessions (
                id TEXT PRIMARY KEY,
                parent_session_id TEXT,
                started_at REAL NOT NULL,
                ended_at REAL
            );
            INSERT INTO sessions(id, parent_session_id, started_at, ended_at)
            VALUES ('foreground-session', NULL, 1.0, 10.0);
            INSERT INTO sessions(id, parent_session_id, started_at, ended_at)
            VALUES ('historical-child-session', 'foreground-session', 2.0, 3.0);
            """
        )
        conn.commit()
        conn.close()

        config = LCMConfig(database_path=str(tmp_path / "lcm_historical_child.db"))
        engine = LCMEngine(config=config, hermes_home=str(hermes_home))
        engine.on_session_start(
            "foreground-session",
            hermes_home=str(hermes_home),
            platform="telegram",
            context_length=200000,
        )
        engine.on_session_start(
            "historical-child-session",
            hermes_home=str(hermes_home),
            platform="telegram",
            context_length=200000,
        )

        assert engine._thread_context_session_id() == ""
        assert not engine._thread_context_has_auxiliary_session("historical-child-session")
        assert engine._session_id == "historical-child-session"
        messages = [
            {"role": "user", "content": "historical child resume should persist normally"},
        ]
        engine.should_compress_preflight(messages)

        assert engine._store.get_session_count("historical-child-session") == 1
