"""#758: a durable redaction policy change must not make every restart re-store the session.

Each stored row holds its text as redacted by the policy that wrote it; a restart redacts the
host's replay with the policy active now. Reconciliation retries the store-tail match with both
sides in one digest-bearing placeholder form, so a replay that is only redacted differently is
proven, while a digest-less (password_assignment) row is still never replay proof (#484 11b).

All secrets are synthetic. No model is called.
"""
from __future__ import annotations

import re

import pytest

from hermes_lcm import ingest_protection as ip
from hermes_lcm import reconcile as lcm_reconcile
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine

SID = "issue-758"
API = "api_key=sk-test1234567890abcdefXYZ"
API_OTHER = "api_key=sk-other0987654321zyxwvUTS"
PWD = "password=hunter2-synthetic-pw"
ENV = 'export OPENAI_API_KEY="sk-proj-abcdefghijklmnop1234"'
NARROW_API = ip._SENSITIVE_PATTERN_CATALOG["api_key"]
# The #491 widening: ``\b`` -> ``(?<![^\W_])`` before the key names.
WIDE_API = re.compile(
    NARROW_API.pattern.replace(r"(?:\\?[\"']?)\b(?:api", r"(?:\\?[\"']?)(?<![^\W_])(?:api", 1),
    NARROW_API.flags,
)
OFF = {"sensitive_patterns_enabled": False}
ON_ALL = {"sensitive_patterns_enabled": True}
ON_API = {"sensitive_patterns_enabled": True, "sensitive_patterns": ["api_key"]}
ON_API_PWD = {"sensitive_patterns_enabled": True, "sensitive_patterns": ["api_key", "password_assignment"]}
ACROSS = "replayed durable tail across a redaction policy change"


def _rows(line: str, *, system: bool = False) -> list[dict]:
    rows = [{"role": "system", "content": "you are a test agent"}] if system else []
    for i in range(10):
        rows.append({"role": "user", "content": f"user turn {i}" + (f"\n{line}" if i == 3 and line else "")})
        rows.append({"role": "assistant", "content": f"assistant reply {i}"})
    return rows


def _engine(tmp_path, monkeypatch, cfg: dict, *, wide: bool = False) -> LCMEngine:
    monkeypatch.setitem(ip._SENSITIVE_PATTERN_CATALOG, "api_key", WIDE_API if wide else NARROW_API)
    config = LCMConfig(
        database_path=str(tmp_path / "lcm.db"),
        large_output_externalization_path=str(tmp_path / "externalized"),
    )
    for key, value in cfg.items():
        setattr(config, key, value)
    engine = LCMEngine(config=config, hermes_home=str(tmp_path / "home"))
    engine.on_session_start(SID, platform="cli", conversation_id=SID, context_length=200_000)
    return engine


def _restarts(tmp_path, monkeypatch, host, cfg1, cfg2, *, restarts=1, wide1=False, wide2=False, replay=None):
    engine = _engine(tmp_path, monkeypatch, cfg1, wide=wide1)
    try:
        engine.ingest(list(host))
        counts = [engine._store.get_session_count(SID)]
    finally:
        engine.shutdown()
    reasons = []
    host = list(replay if replay is not None else host)
    for turn in range(restarts):
        host.append({"role": "user", "content": f"new user row after restart {turn}"})
        engine = _engine(tmp_path, monkeypatch, cfg2, wide=wide2)
        try:
            engine.ingest(list(host))
            counts.append(engine._store.get_session_count(SID))
            reasons.append(engine._last_ingest_reconciliation["reason"])
        finally:
            engine.shutdown()
    return counts, reasons


@pytest.mark.parametrize(
    ("line", "cfg1", "cfg2", "wide2"),
    [
        pytest.param(API, OFF, ON_ALL, False, id="turned-on"),
        pytest.param(ENV, ON_API, ON_API, True, id="api_key-widened-491"),
        pytest.param(API, ON_ALL, OFF, False, id="turned-off", marks=pytest.mark.xfail(strict=True, reason="#758 remaining: no record of the policy a row was written under, so a restart after redaction is turned off or a pattern removed still re-stores")),
        pytest.param(API, ON_API_PWD, {"sensitive_patterns_enabled": True, "sensitive_patterns": ["password_assignment"]},
                     False, id="api_key-removed", marks=pytest.mark.xfail(strict=True, reason="#758 remaining: no record of the policy a row was written under, so a restart after redaction is turned off or a pattern removed still re-stores")),
    ],
)
def test_policy_change_restart_replays_instead_of_restoring(tmp_path, monkeypatch, line, cfg1, cfg2, wide2):
    counts, reasons = _restarts(tmp_path, monkeypatch, _rows(line), cfg1, cfg2, wide2=wide2)
    assert counts == [20, 21]
    assert reasons == [ACROSS]


@pytest.mark.parametrize("system", [False, True])
def test_policy_change_does_not_repeat_on_later_restarts(tmp_path, monkeypatch, system):
    counts, reasons = _restarts(tmp_path, monkeypatch, _rows(API, system=system), OFF, ON_ALL, restarts=3)
    base = 21 if system else 20
    assert counts == [base, base + 1, base + 2, base + 3]  # main: 20 -> 41 -> 63 -> 86
    assert reasons == [ACROSS] * 3


@pytest.mark.parametrize(
    ("line", "cfg1", "cfg2"),
    [pytest.param(API, ON_API, ON_API, id="unchanged-policy"), pytest.param("", OFF, ON_ALL, id="no-matching-row"),
     pytest.param(API, OFF, OFF, id="redaction-off-both-runs")],
)
def test_unaffected_restarts_never_run_the_retry(tmp_path, monkeypatch, line, cfg1, cfg2):
    calls = []
    real = lcm_reconcile.ReconcileMixin._policy_neutral_identity
    monkeypatch.setattr(lcm_reconcile.ReconcileMixin, "_policy_neutral_identity",
                        lambda self, identity: calls.append(1) or real(self, identity))
    counts, reasons = _restarts(tmp_path, monkeypatch, _rows(line), cfg1, cfg2)
    assert counts == [20, 21]
    assert reasons == ["replayed durable tail"]
    assert calls == []


def test_a_different_secret_is_never_proven_by_the_retry(tmp_path, monkeypatch):
    # The host's row 7 carries ANOTHER key than the stored row: not a replay of it.
    replay = _rows(API_OTHER)
    counts, reasons = _restarts(tmp_path, monkeypatch, _rows(API), OFF, ON_ALL, replay=replay)
    assert counts == [20, 41]
    assert reasons == ["persisted ambiguous delta"]


def test_a_new_row_repeating_a_stored_secret_line_is_still_stored(tmp_path, monkeypatch):
    host = _rows(API)
    replay = host + [{"role": "user", "content": f"user turn 3\n{API}"}]
    counts, reasons = _restarts(tmp_path, monkeypatch, host, OFF, ON_ALL, replay=replay)
    assert counts == [20, 22]  # the repeated line and the restart row are both new
    assert reasons == [ACROSS]


def test_digest_less_password_rows_stay_unproven(tmp_path, monkeypatch):
    # #484 item 11b: a password_assignment placeholder is not identity, so the retry leaves it out.
    counts, reasons = _restarts(tmp_path, monkeypatch, _rows(PWD), ON_API, ON_API_PWD)
    assert counts == [20, 41]
    assert reasons == ["persisted ambiguous delta"]
