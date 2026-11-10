"""#2377 (#2414 thread g_8): a replay whose call and/or result carry inline base64 payloads.
Protection mints a fresh ref per ingest, so the tool linker must key on placeholder-restored
identity (as the skip proof does), or the unlinked copy funds a skip of a new occurrence."""
import pytest

from tests.invariants._ingest_shapes_2377 import U, call
from tests.invariants.test_issue_2377_link_write_fault import run, NEXT

pytestmark = pytest.mark.issue(2377)

PAYLOAD = "data:image/png;base64," + "QUJD" * 200
XP = {"role": "tool", "tool_call_id": "c", "content": "result " + PAYLOAD}


def pcall(stamp=None):
    return call(stamp, ids=("c",), args='{"img": "%s"}' % PAYLOAD)


def plan(ordered, where="both"):
    c = pcall if where in ("both", "call") else call
    x = XP if where in ("both", "result") else {"role": "tool", "tool_call_id": "c", "content": "X"}
    pair = [c(), dict(x)]
    first = [U, c(2.0), dict(x, timestamp=2.0)]
    second = [U, c(4.0), dict(x, timestamp=4.0), NEXT]
    third = ([U, *pair, NEXT, *pair, dict(NEXT, content="next2", timestamp=2000.0)] if ordered
             else [U, *pair, *pair, dict(NEXT, content="next2", timestamp=2000.0)])
    return [dict(view=first, cursor=0), dict(view=second, cursor=1), dict(view=third, cursor=1)]


@pytest.mark.parametrize("where", ["both", "call", "result"])
@pytest.mark.parametrize("ordered", [False, True], ids=["adjacent", "after-next"])
def test_payload_replay_copy_never_funds_a_skip(tmp_path, ordered, where):
    stored, log = run(plan(ordered, where), tmp_path / "native", False, "none")
    new = {(2, 4), (2, 5)} if ordered else {(2, 3), (2, 4)}
    conn_rows = log[1]["rows"]
    refs = [r for r in conn_rows if "Externalized" in str(r[2])] or (["call"] if where == "call" else [])
    print("PAYLOAD", where, ordered, sorted(stored), "rels1=", log[1]["relations"], "rels2=", log[2]["relations"], "refs=", len(refs))
    assert refs, "protection never externalized the payload (probe is not exercising the claim)"
    assert all(entry["failures"] == 0 for entry in log), log
    assert all(not entry["unmarked"] for entry in log), log
    assert new <= set(stored), dict(stored=sorted(stored), rows=[r[:2] + (str(r[2])[:40], r[3]) for r in log[2]["rows"]],
                                    rels=log[2]["relations"])
