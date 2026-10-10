"""#1055: effective session windows scale only the non-tool ingest floor."""

from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine


SMALL = 65_536
LARGE = 1_048_576
SESSION = "window-floor"


def _text(n=143_774):
    prose = "This document describes ordinary plans and useful observations. "
    return (prose * (n // len(prose) + 1))[:n]


def _config(tmp_path, **overrides):
    return LCMConfig(**{
        "database_path": str(tmp_path / "lcm.db"),
        "large_output_externalization_enabled": True,
        "large_output_externalization_threshold_chars": 12_000,
        "large_output_active_replay_stubbing_enabled": True,
        "embeddings_enabled": False,
        "summary_model": "stub-model",
        **overrides,
    })


def _engine(tmp_path, window=LARGE, *, config=None, session=SESSION):
    engine = LCMEngine(config=config or _config(tmp_path), hermes_home=str(tmp_path / "hermes"))
    kwargs = {} if window is None else {"context_length": window}
    engine.on_session_start(session, platform="cli", conversation_id=session, **kwargs)
    return engine


def test_143774_char_user_message_stays_inline_at_1m(tmp_path):
    engine = _engine(tmp_path)
    text = _text()
    assert len(text) == 143_774
    engine.ingest([{"role": "user", "content": text}])
    [row] = engine._store.get_session_messages(SESSION)
    assert row["content"] == text
