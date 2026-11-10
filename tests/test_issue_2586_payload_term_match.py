"""#2586: saved outputs match by the message-search term rule, not only the query as typed."""

from types import SimpleNamespace

import json

import pytest

from hermes_lcm import tools
from tests.test_issue_2225_reader_payload_search import Archive, payload_hits

PAD = "padding " * 1700
GOLD = "KAZNIRI-9302"
FACT = f"storm relay evacuation badge: {GOLD}"


@pytest.fixture
def archive(tmp_path):
    archive = Archive(tmp_path / "store")
    yield archive
    archive.close()


def saved(archive, text, session="saved", **kwargs):
    archive.save(session, PAD + "\n" + text + "\n" + PAD, **kwargs)


def found(result, gold=GOLD):
    return [hit for hit in payload_hits(result) if gold in hit["snippet"]]


# The four probe rows from #2225 run 2, then shapes that are not in those questions.
@pytest.mark.parametrize("query", [
    "storm relay", '"storm relay"', "storm relay badge", "storm relay evacuation badge value",
    "BADGE evacuation", 'evacuation "storm relay"', "relay* badge", "storm-relay badge",
    "Storm Relay   Evacuation", '"relay evacuation" storm',
])
def test_a_saved_output_matches_by_terms(archive, query):
    saved(archive, FACT)
    result = archive.call(query=query)
    assert len(found(result)) == 1, result


ROCKET, STORM_CJK = chr(0x1F680), chr(0x5D50)  # emoji and CJK take message search's raw substring route


@pytest.mark.parametrize("query", ["zzabsentterm", '"relay storm"', "storm NOT badge", "relays",
                                   f"storm {ROCKET}", f'"storm {ROCKET}"', f"storm {STORM_CJK}"])
def test_absent_terms_a_reversed_phrase_and_not_never_match(archive, query):
    saved(archive, FACT)
    assert found(archive.call(query=query)) == []


def test_an_explicit_or_query_shows_its_payload_hits_at_once(archive):
    saved(archive, FACT)
    result = archive.call(query="zzabsentterm OR evacuation")
    assert "retried" not in result and len(found(result)) == 1, result


def test_an_underscore_separates_tokens_as_in_the_message_index(archive):
    saved(archive, f"storm_relay evacuation: {GOLD}")
    assert len(found(archive.call(query='"storm relay"'))) == 1


def test_a_combining_mark_joins_a_token_as_in_the_message_index(archive):
    saved(archive, f"storm\u0301relay evacuation: {GOLD}")
    assert found(archive.call(query='"storm relay"')) == []


@pytest.mark.parametrize("text,hit", [("alpha only", False), ("gamma and delta", True), ("beta then alpha", True)])
def test_or_groups_bind_looser_than_and(archive, text, hit):
    saved(archive, f"{text} {GOLD}")
    assert bool(found(archive.call(query="alpha beta OR gamma delta"))) is hit


def test_a_capped_any_term_retry_reports_response_cap(archive, monkeypatch):
    for n in range(3):
        saved(archive, f"storm lantern {n} {GOLD}", session=f"s{n}")
    monkeypatch.setattr(tools, "_LCM_GREP_RESPONSE_CHAR_CAP", 1500)
    result = archive.call(query="storm zzabsentterm")
    assert result["retried"] == "any_term" and 0 < len(payload_hits(result)) < 3, result
    assert "response_cap" in result["partial"]["reasons"]


def test_a_case_fold_that_changes_length_still_matches_by_terms(archive):
    dotted_i = chr(0x130)  # lower() turns it into two characters, so offsets fall back to the plain regex search
    saved(archive, f"{dotted_i}stanbul storm relay evacuation badge: {GOLD}")
    hit, = found(archive.call(query="badge storm"))
    assert hit["snippet"].find("storm") >= 0


def test_an_out_of_scope_payload_never_matches(archive):
    saved(archive, FACT, session="other")
    archive.store.append("saved", {"role": "user", "content": "unrelated"})
    for query in ("storm relay badge", "storm relay evacuation badge value", '"storm relay"'):
        result = archive.call(query=query, session_scope="session", session_id="saved")
        assert payload_hits(result) == [], result


def test_exact_before_all_terms_and_the_snippet_anchors_at_the_first_term(archive):
    exact_text = f"note: storm relay badge {GOLD}"
    saved(archive, exact_text, session="exact", created_at=1.0)
    terms_text = f"badge first, then storm and relay {GOLD}"
    saved(archive, terms_text, session="terms", created_at=2.0)  # newer, but only all terms
    first, second = payload_hits(archive.call(query="storm relay badge"))
    assert first["session_id"] == "exact" and second["session_id"] == "terms"
    content = PAD + "\n" + terms_text
    assert second["byte_position"] == content.index("badge") and second["line"] == 2
    assert second["snippet"].startswith(content[content.index("badge") - 120:][:20])


@pytest.mark.parametrize("query", ['""', '"', '"unbalanced storm', "storm.*relay(", "a" * 10000, "storm " * 2000])
def test_odd_queries_return_without_error(archive, query):
    saved(archive, FACT)
    result = archive.call(query=query)
    assert isinstance(result["results"], list)


def test_a_long_query_keeps_the_literal_match_only(archive):
    saved(archive, FACT)
    many = " ".join(f"word{n}" for n in range(tools._PAYLOAD_TERM_CAP)) + " storm relay"
    assert found(archive.call(query=many)) == []


def test_the_word_cap_counts_the_words_inside_a_phrase(archive):
    saved(archive, FACT)
    phrase = " ".join(f"word{n}" for n in range(tools._PAYLOAD_TERM_CAP))
    assert found(archive.call(query=f'storm "{phrase}"')) == []  # 17 words: literal only, no any-term candidate
    assert len(found(archive.call(query=f'storm "{" ".join(phrase.split()[:14])}"'))) == 1  # 15 words: retry finds storm


def test_the_private_stash_flag_is_not_taken_from_tool_args(archive):
    saved(archive, FACT)
    for mode in ("full_text", "hybrid"):
        raw, _ = archive.reader.call("lcm_grep", {"query": "storm relay evacuation badge value", "mode": mode,
                                                  "_payload_any_term_stash": True})
        assert "_payload_any_term" not in json.loads(raw)


def test_the_any_term_retry_is_capped_and_ordered_by_terms(archive):
    for n in range(12):
        saved(archive, f"storm lantern {n}", session=f"one{n}", created_at=100.0 + n)  # newer, one term each
    saved(archive, f"storm relay ferry {GOLD}", session="three", created_at=1.0)
    result = archive.call(query="storm relay ferry absentword", limit=5)
    assert result["retried"] == "any_term" and len(result["results"]) <= 5
    assert payload_hits(result)[0]["session_id"] == "three"


def test_partial_and_scan_counts_keep_their_meaning(archive):
    saved(archive, FACT)
    result = archive.call(query="storm relay badge")
    scan = result["externalized_scan"]
    assert scan["matched_files"] == 1 and scan["scanned_files"] == 1
    retried = archive.call(query="storm relay evacuation badge value")
    assert retried["externalized_scan"]["matched_files"] == 0 and "_payload_any_term" not in retried


@pytest.mark.parametrize("query", ['"storm relay"', "storm relay badge", "storm relay evacuation badge value"])
def test_the_active_session_mode_matches_by_terms(archive, query):
    saved(archive, FACT)
    engine = SimpleNamespace(_config=archive.config, _hermes_home=str(archive.home),
                             _store=archive.store, _dag=archive.dag, current_session_id="saved")
    result = json.loads(tools.lcm_grep({"query": query, "content_scope": "both"}, engine=engine))
    assert len(found(result)) == 1, result


# Bar 2 (#2587): the str.find fast path returns exactly what the plain regex search returns. Special characters are
# built with chr() so the source carries no escape chains.
SPECIAL = [chr(c) for c in (0x131, 0x130, 0x17F, 0xB5, 0x3BC, 0x3C2, 0x3C3, 0x3A3, 0x212A, 0xDF, 0xFB05, 0xFB06,
                            0x3D0, 0x3B2, 0x1E9B, 0x1E61, 0x345, 0x301)]
FUZZ_ALPHABET = list("isktmeaISKTM .-_") + SPECIAL


def _reference(monkeypatch, *args):
    with monkeypatch.context() as patch:
        patch.setattr(tools, "_ASCII_CASE_FOLDS", __import__("re").compile(r"[\s\S]"))  # every content takes the regex
        return tools._payload_match(*args)


def _differential_cases():
    import random
    rng = random.Random(2587)
    words = ["id", "storm", "sigma", "kelvin", "st", "beta", "mum", "ss",
             chr(0x131) + "d", chr(0x17F) + "torm", chr(0xB5) + "m", chr(0x3BC) + "m", chr(0x3BF) + chr(0x3C2),
             chr(0x3A3) + "IGMA", chr(0x212A) + "elvin", chr(0xFB05), chr(0xFB06), chr(0x3D0) + "eta", "s" + chr(0x301)]
    contents = [f"{a} x {b}" for a in words for b in ("storm", "id", chr(0x3C3))] + [
        "".join(rng.choice(FUZZ_ALPHABET) for _ in range(rng.randint(1, 24))) for _ in range(300)]
    queries = words + [f"{a} {b}" for a, b in zip(words, reversed(words))] + [f'"{a} {b}"' for a, b in zip(words, words[1:])]
    return contents, queries


def test_the_fast_path_equals_the_regex_search(monkeypatch):
    from hermes_lcm.search_query import agent_grep_operands
    contents, queries = _differential_cases()
    compared = 0
    for query in queries:
        _, tokens, operands = agent_grep_operands(query)
        for content in contents:
            for any_term in (False, True):
                assert tools._payload_match(content, query, tokens, operands, any_term) == _reference(
                    monkeypatch, content, query, tokens, operands, any_term), (content, query, any_term)
                compared += 1
    assert compared > 10000


def test_the_fast_path_is_taken_for_plain_ascii_content(monkeypatch):
    calls = []
    real = tools._operand_hit
    monkeypatch.setattr(tools, "_operand_hit", lambda pattern, content, lower, first: calls.append(lower is not None) or real(pattern, content, lower, first))
    tools._payload_match("padding storm relay", "relay storm", ["relay", "storm"], ["relay", "storm"])
    tools._payload_match(chr(0x17F) + " storm relay", "relay storm", ["relay", "storm"], ["relay", "storm"])
    assert calls == [True, True, False, False]
