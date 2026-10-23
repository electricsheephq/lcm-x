"""FIX2 regressions/controls: complete synthetic review tables and deictic safety."""
import json
import re

import pytest

from test_retraction_delivery import add, deliver, detector, links
from test_recall_ranking_contract import corpus as corpus

# Explicit expectations from the approved object/attribution contract; labels compare Phase 2.
CASES = [
    ('control', 'classifier-table-001', 'Forget what I told you about saffron tea.', 'saffron tea', 'synthetic'),
    ('control', 'classifier-table-002', 'Forget what I just told you about saffron tea.', 'saffron tea', 'synthetic'),
    ('control', 'classifier-table-003', 'Please forget what I told you.', '', 'synthetic'),
    ('control', 'classifier-table-004', 'Please forget what I just told you.', '', 'synthetic'),
    ('control', 'classifier-table-005', 'Forget what I told you earlier.', '', 'synthetic'),
    ('control', 'classifier-table-006', 'Forget what I said earlier.', '', 'synthetic'),
    ('control', 'classifier-table-007', 'Forget what I just said.', '', 'synthetic'),
    ('control', 'classifier-table-008', 'Forget what I said.', '', 'synthetic'),
    ('control', 'classifier-table-009', 'Forget what I mentioned.', '', 'synthetic'),
    ('control', 'classifier-table-010', 'forget what i told you', '', 'synthetic'),
    ('control', 'classifier-table-011', 'FORGET WHAT I TOLD YOU ABOUT SAFFRON TEA!', 'saffron tea', 'synthetic'),
    ('control', 'classifier-table-012', 'Forget what I told you about saffron tea!!!', 'saffron tea', 'synthetic'),
    ('control', 'classifier-table-013', 'Forget what I told you about saffron tea?', 'saffron tea', 'synthetic'),
    ('control', 'classifier-table-014', 'Disregard what I told you about the violet launch color.', 'the violet launch color', 'synthetic'),
    ('control', 'classifier-table-015', 'Ignore what I told you about saffron tea.', 'saffron tea', 'synthetic'),
    ('control', 'classifier-table-016', 'Ignore what I said about saffron tea.', 'saffron tea', 'synthetic'),
    ('regression', 'classifier-table-017', 'Disregard what I mentioned regarding saffron tea.', 'saffron tea', 'synthetic'),
    ('control', 'classifier-table-018', 'Forget what I mentioned about saffron tea.', 'saffron tea', 'synthetic'),
    ('control', 'classifier-table-019', 'Forget everything I told you about saffron tea.', 'saffron tea', 'synthetic'),
    ('control', 'classifier-table-020', 'Forget anything I said about saffron tea.', 'saffron tea', 'synthetic'),
    ('control', 'classifier-table-021', 'Forget my earlier statement.', '', 'synthetic'),
    ('control', 'classifier-table-022', 'Forget my earlier statement about saffron tea.', 'saffron tea', 'synthetic'),
    ('control', 'classifier-table-023', 'Forget my previous statement please.', '', 'synthetic'),
    ('control', 'classifier-table-024', 'Disregard my earlier statement.', '', 'synthetic'),
    ('control', 'classifier-table-025', 'Can you please forget what I told you about saffron tea?', 'saffron tea', 'synthetic'),
    ('control', 'classifier-table-026', 'Could you forget what I told you?', '', 'synthetic'),
    ('control', 'classifier-table-027', 'I need you to forget what I told you about saffron tea.', 'saffron tea', 'synthetic'),
    ('control', 'classifier-table-028', 'Would you mind forgetting what I told you about saffron tea?', 'saffron tea', 'synthetic'),
    ('control', 'classifier-table-029', 'Forget that please.', '', 'synthetic'),
    ('control', 'classifier-table-030', 'Forget it now.', '', 'synthetic'),
    ('control', 'classifier-table-031', 'Forget earlier.', '', 'synthetic'),
    ('control', 'classifier-table-032', 'Forget all that.', '', 'synthetic'),
    ('control', 'classifier-table-033', 'Forget that.', '', 'synthetic'),
    ('control', 'classifier-table-034', 'Please forget that, thanks.', 'that, thanks', 'synthetic'),
    ('regression', 'classifier-table-035', 'Forget what I told you, it was wrong.', '', 'synthetic'),
    ('control', 'classifier-table-036', 'Forget what I told you about it.', '', 'synthetic'),
    ('regression', 'classifier-table-037', 'Forget what I told you about saffron tea, it was wrong.', 'saffron tea', 'synthetic'),
    ('control', 'classifier-table-038', 'Forget what I told you about my boss.', 'my boss', 'synthetic'),
    ('control', 'classifier-table-039', 'Forget what I told you about what my boss said.', 'what my boss said', 'synthetic'),
    ('control', 'classifier-table-040', 'Forget what I told you about the tell me game.', None, 'synthetic'),
    ('control', 'classifier-table-041', 'Forget what I told you about the cat who says hi.', None, 'synthetic'),
    ('regression', 'classifier-table-042', 'Forget what I told you ‘about’ saffron tea.', '', 'synthetic'),
    ('control', 'classifier-table-043', 'Forget my favorite drink saffron tea now.', 'my favorite drink saffron tea now', 'synthetic'),
    ('control', 'classifier-table-044', 'Please forget my favorite drink saffron tea.', 'my favorite drink saffron tea', 'synthetic'),
    ('control', 'classifier-table-045', 'Stop using my earlier statement about saffron tea.', 'saffron tea', 'synthetic'),
    ('control', 'classifier-table-046', 'Okay. Forget what I told you about saffron tea.', None, 'synthetic'),
    ('control', 'classifier-table-047', 'Thanks. Please forget what I told you.', None, 'synthetic'),
    ('control', 'classifier-table-048', 'Forget the deadline, my boss told me.', None, 'synthetic'),
    ('control', 'classifier-table-049', 'Forget the deadline, my boss said.', 'the deadline, my boss said', 'synthetic'),
    ('control', 'classifier-table-050', 'Forget it, she told you already.', None, 'synthetic'),
    ('control', 'classifier-table-051', 'Forget what she told you about saffron tea.', None, 'synthetic'),
    ('control', 'classifier-table-052', 'Forget what he said to forget.', None, 'synthetic'),
    ('control', 'classifier-table-053', 'He said to forget it.', None, 'synthetic'),
    ('control', 'classifier-table-054', 'My boss told me to forget the deadline.', None, 'synthetic'),
    ('control', 'classifier-table-055', 'She told you to forget saffron tea.', None, 'synthetic'),
    ('control', 'classifier-table-056', '"Forget what I told you about saffron tea."', None, 'synthetic'),
    ('control', 'classifier-table-057', 'The note says "forget what I told you".', None, 'synthetic'),
    ('control', 'classifier-table-058', '“Forget what I told you about saffron tea.”', None, 'synthetic'),
    ('control', 'classifier-table-059', 'Should I forget what I told you?', None, 'synthetic'),
    ('control', 'classifier-table-060', 'Should you forget what I told you?', None, 'synthetic'),
    ('control', 'classifier-table-061', 'If I ask, forget what I told you.', None, 'synthetic'),
    ('control', 'classifier-table-062', 'Hypothetically, forget what I told you about saffron tea.', None, 'synthetic'),
    ('control', 'classifier-table-063', 'Forget what I told you if I ask again.', None, 'synthetic'),
    ('control', 'classifier-table-064', 'Forget what I told you hypothetically.', None, 'synthetic'),
    ('control', 'classifier-table-065', "Don't forget what I told you.", None, 'synthetic'),
    ('control', 'classifier-table-066', 'Do not forget what I told you about saffron tea.', None, 'synthetic'),
    ('control', 'classifier-table-067', 'Never forget what I told you.', None, 'synthetic'),
    ('control', 'classifier-table-068', 'Forget what I told you, then tell me about music.', None, 'synthetic'),
    ('control', 'classifier-table-069', 'Forget what I said, talk about music instead.', None, 'synthetic'),
    ('regression', 'classifier-table-070', 'Forget what I said to forget.', '', 'synthetic'),
    ('control', 'classifier-table-071', 'Forget what I said to you.', '', 'synthetic'),
    ('control', 'classifier-table-072', 'Forget what I told you to do.', '', 'synthetic'),
    ('control', 'classifier-table-073', 'Forget what I told you about to do lists.', 'to do lists', 'synthetic'),
    ('control', 'classifier-table-074', 'Forget to call mom.', None, 'synthetic'),
    ('control', 'classifier-table-075', 'Forget about it.', None, 'synthetic'),
    ('control', 'classifier-table-076', "Don't forget to book X", None, 'synthetic'),
    ('control', 'classifier-table-077', 'I forgot my keys', None, 'synthetic'),
    ('control', 'classifier-table-078', 'Forget the weather, tell me about Y', None, 'synthetic'),
    ('control', 'classifier-table-079', 'I no longer like X', None, 'synthetic'),
    ('control', 'classifier-table-080', 'If I ever ask you to forget…', None, 'synthetic'),
    ('control', 'classifier-table-081', 'She told her assistant to forget…', None, 'synthetic'),
    ('control', 'classifier-table-082', "Don't forget to book saffron tea.", None, 'synthetic'),
    ('control', 'classifier-table-083', 'I forgot my keys.', None, 'synthetic'),
    ('control', 'classifier-table-084', 'Forget the weather, tell me about music.', None, 'synthetic'),
    ('control', 'classifier-table-085', 'I no longer like saffron tea.', None, 'synthetic'),
    ('control', 'classifier-table-086', 'If I ever ask you to forget saffron tea, ask why.', None, 'synthetic'),
    ('control', 'classifier-table-087', 'She told her assistant to forget saffron tea.', None, 'synthetic'),
    ('control', 'classifier-table-088', 'The note says "please forget saffron tea".', None, 'synthetic'),
    ('control', 'classifier-table-089', '"Please forget saffron tea."', None, 'synthetic'),
    ('control', 'classifier-table-090', 'I am forgetting saffron tea lately.', None, 'synthetic'),
    ('regression', 'classifier-table2-001', 'Forget what I told you, he said.', None, 'synthetic'),
    ('regression', 'classifier-table2-002', 'Forget what I told you, she said.', None, 'synthetic'),
    ('regression', 'classifier-table2-003', 'Forget what I told you, my mom said.', None, 'synthetic'),
    ('regression', 'classifier-table2-004', 'Forget everything I told you, the manager said.', None, 'synthetic'),
    ('regression', 'classifier-table2-005', 'Forget what I told you, he said with a laugh.', None, 'synthetic'),
    ('control', 'classifier-table2-006', 'Forget what I told you, he said to me.', None, 'synthetic'),
    ('control', 'classifier-table2-007', 'Forget what I told you, she says.', None, 'synthetic'),
    ('control', 'classifier-table2-008', 'Forget what I told you, my boss told me.', None, 'synthetic'),
    ('regression', 'classifier-table2-009', 'Ignore what I told you, said my sister.', None, 'synthetic'),
    ('regression', 'classifier-table2-010', 'Disregard what I told you, he wrote.', None, 'synthetic'),
    ('regression', 'classifier-table2-011', 'Forget what I told you, he replied.', None, 'synthetic'),
    ('regression', 'classifier-table2-012', 'Forget what I said, he said.', '', 'synthetic'),
    ('regression', 'classifier-table2-013', 'Forget what I mentioned, she said.', '', 'synthetic'),
    ('control', 'classifier-table2-014', 'Forget what I told you?', '', 'synthetic'),
    ('control', 'classifier-table2-015', 'Forget what I said?', '', 'synthetic'),
    ('regression', 'classifier-table2-016', 'Forget what I told you? No way, I meant it.', '', 'synthetic'),
    ('regression', 'classifier-table2-017', 'Forget what I told you? Why would you?', '', 'synthetic'),
    ('control', 'classifier-table2-018', 'Will you forget what I told you?', '', 'synthetic'),
    ('control', 'classifier-table2-019', 'Never mind, forget what I told you.', None, 'synthetic'),
    ('regression', 'classifier-table2-020', 'Forget what I told you she said.', '', 'synthetic'),
    ('regression', 'classifier-table2-021', 'Forget what I told you he likes.', '', 'synthetic'),
    ('regression', 'classifier-table2-022', 'Forget what I told you in the story.', '', 'synthetic'),
    ('regression', 'classifier-table2-023', 'Forget what I told you, just kidding.', '', 'synthetic'),
    ('regression', 'classifier-table2-024', 'Forget what I told you about saffron tea, he said.', None, 'synthetic'),
    ('regression', 'classifier-table2-025', 'Forget what I told you about saffron tea, she said.', None, 'synthetic'),
    ('regression', 'classifier-table2-026', 'Forget what I told you about the plan, my wife said.', None, 'synthetic'),
    ('regression', 'classifier-table2-027', 'Forget what I said about saffron tea, she said.', 'saffron tea', 'synthetic'),
    ('control', 'classifier-table2-028', 'Forget what I told you, if that is okay.', None, 'synthetic'),
    ('control', 'classifier-table2-029', "Forget what I told you about saffron tea if that's okay.", None, 'synthetic'),
    ('control', 'classifier-table2-030', 'Forget what I told you about saffron tea; my boss told me to say that.', None, 'synthetic'),
    ('regression', 'verify-classifier-tail-table-001', 'Forget what I told you before.', '', 'natural'),
    ('regression', 'verify-classifier-tail-table-002', 'Forget what I told you before.', '', 'engineered'),
    ('regression', 'verify-classifier-tail-table-003', 'Forget what I told you before.', '', 'synthetic'),
    ('regression', 'verify-classifier-tail-table-004', 'Forget what I told you yesterday.', '', 'natural'),
    ('regression', 'verify-classifier-tail-table-005', 'Forget what I told you yesterday.', '', 'engineered'),
    ('regression', 'verify-classifier-tail-table-006', 'Forget what I told you yesterday.', '', 'synthetic'),
    ('regression', 'verify-classifier-tail-table-007', 'Forget what I said yesterday.', '', 'natural'),
    ('regression', 'verify-classifier-tail-table-008', 'Forget what I said yesterday.', '', 'engineered'),
    ('regression', 'verify-classifier-tail-table-009', 'Forget what I said yesterday.', '', 'synthetic'),
    ('regression', 'verify-classifier-tail-table-010', 'Forget what I told you again.', '', 'natural'),
    ('regression', 'verify-classifier-tail-table-011', 'Forget what I told you again.', '', 'engineered'),
    ('regression', 'verify-classifier-tail-table-012', 'Forget what I told you again.', '', 'synthetic'),
    ('regression', 'verify-classifier-tail-table-013', 'Forget what I told you, thanks.', '', 'natural'),
    ('regression', 'verify-classifier-tail-table-014', 'Forget what I told you, thanks.', '', 'engineered'),
    ('regression', 'verify-classifier-tail-table-015', 'Forget what I told you, thanks.', '', 'synthetic'),
    ('regression', 'verify-classifier-tail-table-016', 'Forget what I told you, it was wrong.', '', 'natural'),
    ('regression', 'verify-classifier-tail-table-017', 'Forget what I told you, it was wrong.', '', 'engineered'),
    ('regression', 'verify-classifier-tail-table-018', 'Forget what I told you, it was wrong.', '', 'synthetic'),
    ('regression', 'verify-classifier-tail-table-019', 'Forget what I said, it was wrong.', '', 'natural'),
    ('regression', 'verify-classifier-tail-table-020', 'Forget what I said, it was wrong.', '', 'engineered'),
    ('regression', 'verify-classifier-tail-table-021', 'Forget what I said, it was wrong.', '', 'synthetic'),
    ('regression', 'verify-classifier-tail-table-022', 'Disregard what I mentioned before.', '', 'natural'),
    ('regression', 'verify-classifier-tail-table-023', 'Disregard what I mentioned before.', '', 'engineered'),
    ('regression', 'verify-classifier-tail-table-024', 'Disregard what I mentioned before.', '', 'synthetic'),
    ('regression', 'verify-classifier-tail-table-025', 'Ignore what I said before.', '', 'natural'),
    ('regression', 'verify-classifier-tail-table-026', 'Ignore what I said before.', '', 'engineered'),
    ('regression', 'verify-classifier-tail-table-027', 'Ignore what I said before.', '', 'synthetic'),
    ('regression', 'verify-classifier-tail-table-028', 'Forget what I told you a moment ago.', '', 'natural'),
    ('regression', 'verify-classifier-tail-table-029', 'Forget what I told you a moment ago.', '', 'engineered'),
    ('regression', 'verify-classifier-tail-table-030', 'Forget what I told you a moment ago.', '', 'synthetic'),
    ('regression', 'verify-classifier-tail-table-031', 'Forget what I told you last time.', '', 'natural'),
    ('regression', 'verify-classifier-tail-table-032', 'Forget what I told you last time.', '', 'engineered'),
    ('regression', 'verify-classifier-tail-table-033', 'Forget what I told you last time.', '', 'synthetic'),
    ('control', 'verify-classifier-tail-table-034', 'Forget what I told you earlier.', '', 'natural'),
    ('control', 'verify-classifier-tail-table-035', 'Forget what I told you earlier.', '', 'engineered'),
    ('control', 'verify-classifier-tail-table-036', 'Forget what I told you earlier.', '', 'synthetic'),
    ('control', 'verify-classifier-tail-table-037', 'Please forget what I told you.', '', 'natural'),
    ('control', 'verify-classifier-tail-table-038', 'Please forget what I told you.', '', 'engineered'),
    ('control', 'verify-classifier-tail-table-039', 'Please forget what I told you.', '', 'synthetic'),
    ('control', 'verify-classifier-tail-table-040', 'Forget what I said.', '', 'natural'),
    ('control', 'verify-classifier-tail-table-041', 'Forget what I said.', '', 'engineered'),
    ('control', 'verify-classifier-tail-table-042', 'Forget what I said.', '', 'synthetic'),
    ('control', 'verify-classifier-tail-table-043', 'Forget what I just said.', '', 'natural'),
    ('control', 'verify-classifier-tail-table-044', 'Forget what I just said.', '', 'engineered'),
    ('control', 'verify-classifier-tail-table-045', 'Forget what I just said.', '', 'synthetic'),
    ('control', 'verify-classifier-tail-table-046', 'Forget my earlier statement.', '', 'natural'),
    ('control', 'verify-classifier-tail-table-047', 'Forget my earlier statement.', '', 'engineered'),
    ('control', 'verify-classifier-tail-table-048', 'Forget my earlier statement.', '', 'synthetic'),
    ('control', 'verify-classifier-tail-table-049', 'Forget what I told you about the launch color.', 'the launch color', 'natural'),
    ('control', 'verify-classifier-tail-table-050', 'Forget what I told you about the launch color.', 'the launch color', 'engineered'),
    ('control', 'verify-classifier-tail-table-051', 'Forget what I told you about the launch color.', 'the launch color', 'synthetic'),
    ('regression', 'explicit-001', 'Forget what I told you before.', '', 'synthetic'),
    ('regression', 'explicit-002', 'Forget what I told you yesterday.', '', 'synthetic'),
    ('regression', 'explicit-003', 'Forget what I told you again.', '', 'synthetic'),
    ('regression', 'explicit-004', 'Forget what I told you a moment ago.', '', 'synthetic'),
    ('regression', 'explicit-005', 'Forget what I told you last time.', '', 'synthetic'),
    ('regression', 'explicit-006', 'Forget what I said yesterday.', '', 'synthetic'),
    ('regression', 'explicit-007', 'Disregard what I mentioned before.', '', 'synthetic'),
    ('regression', 'explicit-008', 'Ignore what I said before.', '', 'synthetic'),
    ('regression', 'explicit-009', 'Forget what I told you about saffron tea and use coffee.', 'saffron tea', 'synthetic'),
    ('regression', 'explicit-010', 'Forget what I told you regarding saffron tea; it was wrong.', 'saffron tea', 'synthetic'),
    ('regression', 'explicit-011', 'Forget what I mentioned concerning saffron tea. Use coffee.', 'saffron tea', 'synthetic'),
    ('regression', 'explicit-012', 'Disregard my previous information concerning saffron tea, thanks.', 'saffron tea', 'synthetic'),
    ('control', 'explicit-013', 'Forget saffron tea, it was wrong.', 'saffron tea, it was wrong', 'synthetic'),
    ('control', 'explicit-014', 'Stop using saffron tea and coffee.', 'saffron tea and coffee', 'synthetic'),
    ('control', 'explicit-015', 'Okay. Forget what I told you.', None, 'synthetic'),
    ('regression', 'explicit-016', 'Forget what I told you concerning saffron tea.', 'saffron tea', 'synthetic'),
    ('regression', 'explicit-017', 'Forget what I said about saffron tea, it was wrong.', 'saffron tea', 'synthetic'),
    ('regression', 'explicit-018', 'Forget what I said regarding saffron tea.', 'saffron tea', 'synthetic'),
    ('control', 'explicit-019', "Forget what I said about saffron tea if that's okay.", None, 'synthetic'),
    ('control', 'explicit-020', 'Forget saffron tea.', 'saffron tea', 'synthetic'),
    ('regression', 'explicit-021', 'Disregard my earlier statement regarding saffron tea.', 'saffron tea', 'synthetic'),
    ('regression', 'explicit-022', 'Forget what I told you, he said.', None, 'synthetic'),
    ('regression', 'explicit-023', 'Forget what I just told you about saffron tea, he said.', None, 'synthetic'),
    ('regression', 'explicit-024', 'Forget what I told you, she said.', None, 'synthetic'),
    ('regression', 'explicit-025', 'Forget what I just told you about saffron tea, she said.', None, 'synthetic'),
    ('regression', 'explicit-026', 'Forget what I told you, they said.', None, 'synthetic'),
    ('regression', 'explicit-027', 'Forget what I just told you about saffron tea, they said.', None, 'synthetic'),
    ('regression', 'explicit-028', 'Forget what I told you, my mom said.', None, 'synthetic'),
    ('regression', 'explicit-029', 'Forget what I just told you about saffron tea, my mom said.', None, 'synthetic'),
    ('regression', 'explicit-030', 'Forget what I told you, the manager said.', None, 'synthetic'),
    ('regression', 'explicit-031', 'Forget what I just told you about saffron tea, the manager said.', None, 'synthetic'),
    ('regression', 'explicit-032', 'Forget what I told you, he wrote.', None, 'synthetic'),
    ('regression', 'explicit-033', 'Forget what I just told you about saffron tea, he wrote.', None, 'synthetic'),
    ('regression', 'explicit-034', 'Forget what I told you, he replied.', None, 'synthetic'),
    ('regression', 'explicit-035', 'Forget what I just told you about saffron tea, he replied.', None, 'synthetic'),
    ('regression', 'explicit-036', 'Forget what I told you, said my sister.', None, 'synthetic'),
    ('regression', 'explicit-037', 'Forget what I just told you about saffron tea, said my sister.', None, 'synthetic'),
]


@pytest.mark.parametrize("label,case_id,text,expected,decoy_kind", CASES,
                         ids=[f"{row[0]}-{row[1]}" for row in CASES])
def test_fix2_review_table_objects(label, case_id, text, expected, decoy_kind):
    assert detector().request_object(text) == expected


@pytest.mark.parametrize("label,case_id,text,expected,decoy_kind", CASES,
                         ids=[f"{row[0]}-{row[1]}" for row in CASES])
def test_fix2_suppress_decoy_safety(corpus, monkeypatch, label, case_id, text, expected, decoy_kind):
    # The decoy contains tail words, never the explicit object's words.
    tail = re.split(r"(?:told you|mentioned|said|statement|information|details)\b", text, maxsplit=1, flags=re.I)[-1]
    tail = re.split(r"\b(?:about|regarding|concerning)\s+", tail, maxsplit=1, flags=re.I)[0]
    decoy_text = "Archived note: " + tail if tail != text else "Call the plumber before noon."
    if decoy_kind == "engineered":
        decoy_text = "The wrong key was there yesterday before the last moment again, thanks."
    decoy = add(corpus, decoy_text)
    target = add(corpus, f"The record includes {expected}." if expected else "The launch color is violet.")
    assistant = add(corpus, "Acknowledged the record.", "assistant")
    foreign = add(corpus, "Acknowledged elsewhere.", "assistant", session="foreign")
    request = add(corpus, text)
    hits = [decoy, target, assistant, foreign]
    groups, reason = links(corpus, hits)
    assert reason is None
    payload = json.loads(deliver(corpus, monkeypatch, hits, "suppress", limit=4))
    survivors = {hit.get("store_id") for hit in payload["hits"]}
    assert decoy["store_id"] in survivors, "unrelated tail-word decoy was withdrawn"
    assert foreign["store_id"] in survivors
    if expected is None:
        assert groups == []
        assert deliver(corpus, monkeypatch, hits, "suppress", limit=4) == deliver(corpus, monkeypatch, hits, "off", limit=4)
    else:
        assert len(groups) == 1 and groups[0][0]["store_id"] == request["store_id"]
        allowed = {1, 2} if expected == "" else {1}
        assert set(groups[0][1]) == allowed
        assert len(groups[0][1]) <= 3
        assert target["store_id"] not in survivors
        assert any(hit.get("kind") == "retraction" for hit in payload["hits"])


def test_fix2_regression_deictic_three_original_bound(corpus, monkeypatch):
    decoy = add(corpus, "The parcel arrived yesterday.")
    target = add(corpus, "The launch color is violet.")
    assistants = [add(corpus, f"Acknowledged number {i}.", "assistant") for i in range(4)]
    add(corpus, "Forget what I told you yesterday.")
    hits = [decoy, target, *assistants]
    groups, reason = links(corpus, hits)
    assert reason is None and len(groups) == 1
    assert groups[0][1] == [5, 4, 3]
    payload = json.loads(deliver(corpus, monkeypatch, hits, "suppress", limit=6))
    assert {decoy["store_id"], target["store_id"], assistants[0]["store_id"]} <= {
        hit.get("store_id") for hit in payload["hits"]}


# F2 bonus: observe only; no recall target or tuning gate.
F2_TEXTS = ['Forget what I told you about saffron tea.', 'Ignore what I said about saffron tea.', 'Forget what I mentioned about saffron tea.', 'Disregard what I mentioned regarding saffron tea.', 'Forget what I said regarding saffron tea.', 'Forget what I told you concerning saffron tea.', 'Forget what I said about saffron tea, it was wrong.', 'Forget what I told you about saffron tea, it was wrong.', "Forget what I told you about saffron tea if that's okay.", "Forget what I said about saffron tea if that's okay.", 'Forget saffron tea.', 'Disregard my earlier statement regarding saffron tea.']


@pytest.mark.parametrize("text", F2_TEXTS)
def test_fix2_f2_count_only(corpus, monkeypatch, text):
    original = add(corpus, "Saffron tea is my favorite drink.")
    add(corpus, "Unrelated filler about the weather.")
    add(corpus, text)
    groups, reason = links(corpus, [original])
    payload = json.loads(deliver(corpus, monkeypatch, [original], "suppress"))
    assert reason is None
    print("FIX2_F2:" + json.dumps(dict(text=text, obj=detector().request_object(text),
                                      linked=bool(groups), withdrawn=any(
                                          hit.get("kind") == "retraction" for hit in payload["hits"]))))
