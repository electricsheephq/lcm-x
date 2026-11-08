"""#2431 item 4, a named surplus limit (v0.30.0): a retained prompt the host shows with a leading space is stored again.

The prompt is a model-switch note plus P, stored at its stamp after a compaction. After the rebind the host shows it
at the same stamp with a leading space, and the carried turn T is re-stamped. Main stores the leading-space prompt
again as a new row and stores T's second copy with no ``possible_replay_of`` link: two silent copies, nothing lost.
The strict xfail turns green with the fix (v0.30.x) and must then become an ordinary test. The control shows the
same bytes without the leading space: T's copy is linked and the prompt is not stored again.
"""
from copy import deepcopy

import pytest

from tests.invariants.test_issue_2378_restamp_link import (  # noqa: F401 (the build fixture)
    T, T_COMMIT, T_REQ, all_rows, build, history, replay_links)
from tests.invariants.test_issue_2378_stamp_transfer import NEW, NOTE, P

pytestmark = pytest.mark.issue(2431)
PROMPT = NOTE + P


def _restamp_with_retained_prompt(build, cold, shown):  # noqa: F811
    engine = build()
    before = history(False) + [{"role": "user", "content": T, "timestamp": T_REQ}]
    engine.ingest(deepcopy(before))
    view = engine.compress(deepcopy(before), force=True)
    if engine._last_compression_status != "compacted":
        pytest.fail("setup: the compaction did not publish")
    view += [{"role": "assistant", "content": "compress finished", "timestamp": T_REQ + 1},
             {"role": "user", "content": PROMPT, "timestamp": T_REQ}]
    engine.ingest(deepcopy(view))
    engine.on_session_end("S", deepcopy(view))
    if cold:
        engine.shutdown()
        engine = build()
    else:
        engine.on_session_start("S", conversation_id="conversation", platform="cli")
    next(row for row in view if row.get("content") == T)["timestamp"] = T_COMMIT - 1
    next(row for row in view if row.get("content") == PROMPT)["content"] = shown
    view.append({"role": "user", "content": NEW, "timestamp": T_COMMIT + 10})
    engine.ingest(deepcopy(view))
    rows, links = all_rows(engine), replay_links(engine)
    if sum(row["content"] == NEW for row in rows) != 1:
        pytest.fail("setup: the new turn must be stored once")
    t_copies = [row["store_id"] for row in rows if row["content"] == T]
    prompts = [row["store_id"] for row in rows if row["content"].strip() == PROMPT.strip()]
    if len(t_copies) != 2:
        pytest.fail(f"not the #2431 shape: T copies {t_copies}")
    return t_copies, prompts, links


@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason="#2431 named surplus limit (v0.30.0): a retained prompt shown with a leading space is "
                          "stored again, and the re-stamped carried turn gets no link")
@pytest.mark.parametrize("cold", [False, True], ids=["warm", "cold"])
def test_a_leading_space_retained_prompt_is_not_a_silent_copy(build, cold):  # noqa: F811
    t_copies, prompts, links = _restamp_with_retained_prompt(build, cold, " " + PROMPT)
    if (t_copies[1], t_copies[0]) not in links and not (len(prompts) == 2 and links == set()):
        pytest.fail(f"not the #2431 shape: prompts {prompts}, relations {sorted(links)}")
    assert (t_copies[1], t_copies[0]) in links, f"#2431 known shape: T unlinked, prompt copies {prompts}"
    assert len(prompts) == 1 or (prompts[1], prompts[0]) in links, f"#2431 known shape: prompt copies {prompts}"


@pytest.mark.parametrize("cold", [False, True], ids=["warm", "cold"])
def test_the_same_retained_prompt_without_the_space_links_the_restamped_turn(build, cold):  # noqa: F811
    """Control (passes on main)."""
    t_copies, prompts, links = _restamp_with_retained_prompt(build, cold, PROMPT)
    assert (t_copies[1], t_copies[0]) in links
    assert len(prompts) == 1
