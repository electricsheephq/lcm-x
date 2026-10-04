"""Phase C's transcript integrity and stored-row lossless gate; synthetic inputs only."""
import hashlib
import json
import sqlite3

import pytest

from bench.instruments.reliability.scorers import cli, multiset


def run_dir(tmp_path, prompts=("prompt",), answers=("answer",), hashes=True):
    (tmp_path / "actual-material.jsonl").write_text(
        "".join(json.dumps({"text": p}) + "\n" for p in prompts))
    (tmp_path / "probes.jsonl").write_text("")
    (tmp_path / "results.jsonl").write_text("".join(json.dumps({
        "raw_answer": a, **({"input_sha256": hashlib.sha256(prompts[i].encode()).hexdigest()}
                           if hashes and i < len(prompts) else {})}) + "\n" for i, a in enumerate(answers)))
    return tmp_path


@pytest.mark.parametrize("answers", [(), ("a", "b")])
def test_result_count_mismatch_is_payload_free(tmp_path, answers):
    with pytest.raises(ValueError, match="prompt/result count mismatch") as exc:
        cli.gauntlet_transcript(run_dir(tmp_path, prompts=("UNIQUE_PRIVATE_PROMPT_PAYLOAD",), answers=answers))
    assert "UNIQUE_PRIVATE_PROMPT_PAYLOAD" not in str(exc.value)


def test_hash_uses_unstripped_utf8_prompt(tmp_path):
    prompt = " \té\n"
    run_dir(tmp_path, prompts=(prompt,))
    assert cli.gauntlet_transcript(tmp_path)[0] == ("user", prompt)
    (tmp_path / "results.jsonl").write_text(json.dumps({
        "raw_answer": "answer", "input_sha256": hashlib.sha256(prompt.strip().encode()).hexdigest()}))
    with pytest.raises(ValueError, match="input_sha256 mismatch at index 0"):
        cli.gauntlet_transcript(tmp_path)


def test_old_result_without_hash_still_aligns(tmp_path):
    assert cli.gauntlet_transcript(run_dir(tmp_path, hashes=False)) == [
        ("user", "prompt"), ("assistant", "answer")]


def test_cli_count_mismatch_is_inconclusive(tmp_path, capsys, monkeypatch):
    run_dir(tmp_path, answers=())
    db = tmp_path / "lcm.db"
    with sqlite3.connect(db) as con:
        con.execute("create table messages (store_id, session_id, role, content, conversation_id)")
    monkeypatch.setattr(cli.dupes, "count", lambda _: {})
    assert cli.main(["--db", str(db), "--gauntlet-run", str(tmp_path)]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["multiset"]["verdict"] == "INCONCLUSIVE"
    assert "prompt/result count mismatch" in out["multiset"]["reason"]


def test_role_dependent_edges_for_both_callers():
    expected = [("user", " prompt \n"), ("assistant", " answer \n")]
    rows = [(1, "S", "user", "prompt", 1), (2, "S", "assistant", " answer \n", 1)]
    assert multiset.score(expected, [r[:4] for r in rows])["verdict"] == "PASS"
    assert multiset.phase_c_score(expected, rows)["verdict"] == "PASS"
    rows[1] = (2, "S", "assistant", "answer", 1)
    for out in (multiset.score(expected, [r[:4] for r in rows]), multiset.phase_c_score(expected, rows)):
        assert (out["verdict"], out["deficit_rows"], out["surplus_rows"]) == ("FAIL", 1, 1)


def test_row_order_reports_first_transcript_index():
    expected = [("user", "prompt"), ("assistant", "answer")]
    rows = [(1, "S", "assistant", "answer", 1), (2, "S", "user", "prompt", 1)]
    out = multiset.phase_c_score(expected, rows)
    assert out["verdict"] == "FAIL"
    assert out["out_of_order"] == {"transcript_index": 1, "role": "assistant"}


def test_rotation_children_match_only_owning_conversation():
    expected = [("user", "prompt"), ("assistant", "answer")]
    rows = [(1, "root", "user", "prompt", 1), (2, "child", "assistant", "answer", 1),
            (3, "foreign", "assistant", "answer", 2)]
    out = multiset.phase_c_score(expected, rows)
    assert out["verdict"] == "PASS"
    assert out["owning_conversation"] == 1
    assert out["foreign_conversations"] == {2: 1}
    out = multiset.phase_c_score(expected, [rows[0], rows[2]])
    assert (out["verdict"], out["deficit_rows"]) == ("FAIL", 1)
    assert out["foreign_conversations"] == {2: 1}


def test_first_item_missing_is_inconclusive():
    out = multiset.phase_c_score([("user", "prompt")], [(1, "S", "user", "other", 1)])
    assert out["verdict"] == "INCONCLUSIVE"


def split_rows(parts=("one", "two", "three")):
    return [(1, "S", "user", "prompt", 1)] + [
        (i + 2, "S", "assistant", p, 1) for i, p in enumerate(parts)]


def test_split_with_intervening_tools_and_empty_assistant():
    rows = split_rows()
    rows += [(2.5, "S", "tool", "result", 1), (3.5, "S", "assistant", "", 1)]
    out = multiset.phase_c_score([("user", "prompt"), ("assistant", "one\n\ntwo\n\nthree")], rows)
    assert out["verdict"] == "PASS"
    assert out["split_assistant_turns"][0]["split_match"] == [2, 3, 4]
    assert multiset.score([("user", "prompt"), ("assistant", "one\n\ntwo\n\nthree")],
                          [r[:4] for r in rows])["verdict"] == "FAIL"


@pytest.mark.parametrize("deleted", [2, 3, 4])
def test_split_deletion_fails(deleted):
    rows = [r for r in split_rows() if r[0] != deleted]
    assert multiset.phase_c_score([("user", "prompt"), ("assistant", "one\n\ntwo\n\nthree")], rows)[
        "verdict"] == "FAIL"


@pytest.mark.parametrize("boundary", ["", " ", "new prompt"])
def test_every_user_row_is_split_turn_boundary(boundary):
    rows = split_rows() + [(2.5, "S", "user", boundary, 1)]
    assert multiset.phase_c_score([("user", "prompt"), ("assistant", "one\n\ntwo\n\nthree")], rows)[
        "verdict"] == "FAIL"


def test_claimed_fragment_cannot_be_borrowed():
    expected = [("user", "prompt"), ("assistant", "one\n\ntwo\n\nthree"), ("assistant", "two")]
    assert multiset.phase_c_score(expected, split_rows())["verdict"] == "FAIL"


def test_duplicate_used_fragment_fails():
    rows = split_rows() + [(5, "S", "assistant", "three", 1)]
    assert multiset.phase_c_score([("user", "prompt"), ("assistant", "one\n\ntwo\n\nthree")], rows)[
        "verdict"] == "FAIL"


def test_ambiguous_run_fails_before_fragment_use():
    rows = split_rows() + [(i + 5, "other-session", "assistant", p, 1)
                           for i, p in enumerate(("one", "two", "three"))]
    assert multiset.phase_c_score([("user", "prompt"), ("assistant", "one\n\ntwo\n\nthree")], rows)[
        "verdict"] == "FAIL"


@pytest.mark.parametrize("n,verdict", [(2, "PASS"), (8, "PASS"), (9, "FAIL")])
def test_split_fragment_limit(n, verdict):
    parts = tuple(f"part{i}" for i in range(n))
    assert multiset.phase_c_score([("user", "prompt"), ("assistant", "\n\n".join(parts))], split_rows(parts))[
        "verdict"] == verdict


def test_repeated_missing_answer_cannot_use_split():
    expected = [("user", "prompt"), ("assistant", "one\n\ntwo\n\nthree"),
                ("assistant", "one\n\ntwo\n\nthree")]
    assert multiset.phase_c_score(expected, split_rows())["verdict"] == "FAIL"


def test_split_cannot_cross_sessions_or_reverse_fragments():
    expected = [("user", "prompt"), ("assistant", "one\n\ntwo\n\nthree")]
    rows = split_rows()
    rows[-1] = (4, "child", "assistant", "three", 1)
    assert multiset.phase_c_score(expected, rows)["verdict"] == "FAIL"
    assert multiset.phase_c_score(expected, split_rows(("three", "two", "one")))["verdict"] == "FAIL"


@pytest.mark.parametrize("answer,parts", [
    ("onetwothree", ("one", "two", "three")),
    ("one \t two\r\nthree", ("one", "two", "three")),
    ("é two", ("e\u0301", "two")),
    (" one two ", ("one", "two")),
])
def test_inherited_v2_normalized_joins_are_phase_c_only(answer, parts):
    expected = [("user", "prompt"), ("assistant", answer)]
    rows = split_rows(parts)
    assert multiset.phase_c_score(expected, rows)["verdict"] == "PASS"
    assert multiset.score(expected, [r[:4] for r in rows])["verdict"] == "FAIL"


def test_normalized_whole_row_blocks_split_fallback():
    rows = split_rows() + [(5, "S", "assistant", "one two three", 1)]
    out = multiset.phase_c_score([("user", "prompt"), ("assistant", "one\n\ntwo\n\nthree")], rows)
    assert out["verdict"] == "FAIL"
    assert out["accepted_split_keys"] == 0


def test_fragments_cannot_be_reused_between_answers():
    expected = [("user", "prompt"), ("assistant", "one\n\ntwo"), ("assistant", "two\n\nthree")]
    out = multiset.phase_c_score(expected, split_rows())
    assert out["verdict"] == "FAIL"
    assert out["accepted_split_keys"] == 1


def test_ambiguous_first_conversation_is_inconclusive():
    rows = [(1, "S", "user", "prompt", 1), (2, "F", "user", "prompt", 2)]
    assert multiset.phase_c_score([("user", "prompt")], rows)["verdict"] == "INCONCLUSIVE"
