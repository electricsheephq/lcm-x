"""#2118 named surplus limit (v0.30.0): a re-formed carrier the host commits in place is stored after a restart.

The shape: a published compaction and the in-place boundary, 12 more turns, then a held pass whose survival fit
re-forms the carrier around the first kept user row U. The host commits that list in place (``on_session_end`` and
the boundary, or the boundary alone) and restarts before LCM-X's next ingest. The cold engine's first ingest of the
fitted list, the reply and a new turn stores the carrier (the summary text plus U's text) as a new plain user row
with no stamp and no relation. The strict xfail turns green with the fix (v0.30.x) and must then become an
ordinary test. The guard: when a warm ingest claims the fitted list before the restart, nothing is stored again.
"""

import pytest

from tests.test_issue_650_survival_fit_keeps_summary import (  # noqa: F401 (the host fixture)
    _BOUNDARY, _engine, _held_fit_after_inplace_boundary, _rows, host)

pytestmark = pytest.mark.issue(2118)

REPLY = {"role": "assistant", "content": "reply CUR", "timestamp": 30_001.0}
NEW = [{"role": "user", "content": "[NEW] next turn", "timestamp": 50_000.0},
       {"role": "assistant", "content": "reply NEW", "timestamp": 50_001.0}]


def _restart_after_committed_fit(tmp_path, host, monkeypatch, commit, stamp, warm):  # noqa: F811
    try:  # the imported setup helper asserts; its failure must not read as the expected xfail
        engine, fitted, rest = _held_fit_after_inplace_boundary(tmp_path, host, monkeypatch)
    except AssertionError as error:
        pytest.fail(f"setup: {error}")
    if stamp == "host-stamped":
        fitted[0]["timestamp"] = 40_000.0  # the host stamps the carrier object it adopted
    try:
        if commit == "end+boundary":
            engine.on_session_end("S", fitted)
            engine.on_session_start("S", **_BOUNDARY)
        if warm:
            engine.ingest([*fitted, REPLY])
        stored = len(_rows(engine))
    finally:
        engine.shutdown()
    cold = _engine(tmp_path)
    try:
        cold.ingest([*fitted, REPLY, *NEW])
        if cold._ingest_failure_count:
            pytest.fail(f"setup: ingest failed: {cold._last_ingest_error}")
        rows = _rows(cold)
    finally:
        cold.shutdown()
    if sum(1 for row in rows if row["content"] == rest) != 1:
        pytest.fail("not the #2118 shape: U itself is not stored exactly once")
    added = rows[stored:]
    plain = [row["content"] for row in added if "Summary (d" not in str(row["content"])[:40]]
    expected = [NEW[0]["content"], NEW[1]["content"]] if warm else [REPLY["content"], NEW[0]["content"],
                                                                    NEW[1]["content"]]
    if plain != expected:
        pytest.fail(f"not the #2118 shape: plain rows added {[str(c)[:40] for c in plain]}")
    return [row for row in added if "Summary (d" in str(row["content"])[:40]], rest


@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason="#2118 named surplus limit (v0.30.0): a restart before the next ingest stores the "
                          "re-formed carrier the host committed in place as a new unmarked user row")
@pytest.mark.parametrize("stamp", ["unstamped", "host-stamped"])
@pytest.mark.parametrize("commit", ["end+boundary", "boundary-only"])
def test_a_committed_refit_carrier_is_not_stored_after_a_restart(tmp_path, host, monkeypatch, commit, stamp):  # noqa: F811
    carriers, rest = _restart_after_committed_fit(tmp_path, host, monkeypatch, commit, stamp, warm=False)
    stamped = 40_000.0 if stamp == "host-stamped" else None
    if len(carriers) > 1 or any(not str(row["content"]).endswith(rest) or row["observed_at"] != stamped
                                for row in carriers):
        pytest.fail(f"not the #2118 shape: {[(len(str(r['content'])), r['observed_at']) for r in carriers]}")
    assert carriers == [], "#2118 known shape: one carrier row (the summary plus U's text) stored as a new user row"


@pytest.mark.parametrize("stamp", ["unstamped", "host-stamped"])
@pytest.mark.parametrize("commit", ["end+boundary", "boundary-only"])
def test_a_refit_carrier_claimed_before_the_restart_is_not_stored(tmp_path, host, monkeypatch, commit, stamp):  # noqa: F811
    """Guard (passes on main): a warm ingest of the fitted list before the restart; the restart stores only the
    new turn."""
    carriers, _rest = _restart_after_committed_fit(tmp_path, host, monkeypatch, commit, stamp, warm=True)
    assert carriers == []
