from __future__ import annotations

from pathlib import Path

import pytest

from access_context.fixtures import FIXTURE_KINDS, fixture_paths, load_fixture

def test_fixture_corpus_is_non_empty_in_every_kind() -> None:
    minimums = {"positive": 5, "negative": 16, "delegation": 10, "revocation": 4}
    for kind in FIXTURE_KINDS:
        paths = fixture_paths(kind)
        assert len(paths) >= minimums[kind], kind


@pytest.mark.parametrize("path", fixture_paths())
def test_each_discovered_fixture_has_an_envelope(path: Path) -> None:
    payload = load_fixture(path)
    assert payload["contract_revision"]
    assert payload["description"].count("\n") == 0
    assert "context" in payload
    if path.parent.name in {"negative", "delegation", "revocation"}:
        assert "expected" in payload


def test_fixture_kinds_are_closed_and_shared_root_is_present() -> None:
    assert FIXTURE_KINDS == {"positive", "negative", "delegation", "revocation"}
    assert fixture_paths()[0].parents[1].name == "access_context_v1"
