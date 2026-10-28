"""Default-faithful engine construction and one host model-call cycle."""

from contextlib import contextmanager
from dataclasses import fields
from typing import NamedTuple

from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from hermes_lcm.tokens import count_messages_tokens


def make_engine(tmp_path, *, window=None, session_id="S", conversation_id=None,
                platform="cli", **overrides) -> LCMEngine:
    """Use shipped defaults, isolated storage, and explicit config overrides only."""
    if "database_path" in overrides:
        raise TypeError("make_engine always stores in tmp_path/lcm.db; database_path cannot be overridden")
    config = LCMConfig()
    config.database_path = str(tmp_path / "lcm.db")
    names = {field.name for field in fields(config)}
    for name, value in overrides.items():
        if name not in names:
            raise AttributeError(f"Unknown LCMConfig attribute: {name}")
        setattr(config, name, value)
    engine = LCMEngine(config=config, hermes_home=str(tmp_path / "home"))
    if window is not None:
        kwargs = {"platform": platform, "context_length": window}
        if conversation_id is not None:
            kwargs["conversation_id"] = conversation_id
        try:
            engine.on_session_start(session_id, **kwargs)
        except BaseException:
            engine.shutdown()
            raise
    return engine


@contextmanager
def engine_context(tmp_path, *, window=None, session_id="S", conversation_id=None,
                   platform="cli", **overrides):
    """Construct an engine and shut it down even when the test raises."""
    engine = make_engine(tmp_path, window=window, session_id=session_id,
                         conversation_id=conversation_id, platform=platform, **overrides)
    try:
        yield engine
    finally:
        engine.shutdown()


class HostCall(NamedTuple):
    view: list[dict]
    sent_tokens: int
    compressed: bool
    retried: bool
    overflow_failed: bool | None


def host_call(engine, view, *, retry_on_overflow=True) -> HostCall:
    """Ingest, check pressure (the threshold, then the engine's preflight gate), compress, and retry overflow once.

    ``compressed`` records an invocation, including a real engine no-op.
    An unbound/unknown window (zero) does not imply overflow.
    """
    engine.ingest(view)
    n = count_messages_tokens(view)
    compressed = engine.should_compress(n)
    if not compressed:
        # The host's sub-threshold branch: an unblocked request asks the engine's preflight gate, which can
        # request one compress pass (Hermes agent/turn_context_compaction.py, _engine_preflight_maintenance).
        info, preflight = getattr(engine, "should_compress_info", None), getattr(engine, "should_compress_preflight", None)
        blocked = callable(info) and info(n)[1]
        compressed = bool(not blocked and callable(preflight) and preflight(view))
    # Every automatic compress passes the host's breaker gate, read on the type (Hermes
    # agent/conversation_compression.py, _automatic_compression_gate_blocks): a no-progress or sweep hold skips it.
    gate = getattr(type(engine), "_automatic_compression_blocked", None)
    if compressed and callable(gate) and gate(engine):
        compressed = False
    if compressed:
        view = engine.compress(view, current_tokens=n)
        n = count_messages_tokens(view)
    retried = bool(retry_on_overflow and engine.context_length > 0 and n > engine.context_length)
    if retried:
        view = engine.compress(view, current_tokens=n, bypass_cooldown=True)
        n = count_messages_tokens(view)
        compressed = True
    return HostCall(view, n, compressed, retried,
                    getattr(engine, "_last_overflow_recovery_failed", None))


def stored_rows(engine, session_id=None):
    """Read durable rows through MessageStore's public session accessor."""
    if session_id is None:
        session_id = engine.current_session_id
    return engine._store.get_session_messages(session_id)
