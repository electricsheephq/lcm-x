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


@pytest.mark.parametrize("query", ["zzabsentterm", '"relay storm"', "storm NOT badge", "relays"])
def test_absent_terms_a_reversed_phrase_and_not_never_match(archive, query):
    saved(archive, FACT)
    assert found(archive.call(query=query)) == []


def test_an_explicit_or_query_shows_its_payload_hits_at_once(archive):
    saved(archive, FACT)
    result = archive.call(query="zzabsentterm OR evacuation")
    assert "retried" not in result and len(found(result)) == 1, result


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
