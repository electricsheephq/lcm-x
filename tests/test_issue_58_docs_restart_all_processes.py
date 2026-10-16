from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS = ("README.md", "docs/operator-guide.md")


@pytest.mark.parametrize("doc_path", DOCS)
def test_update_section_says_restart_every_long_running_process(doc_path):
    text = (REPO_ROOT / doc_path).read_text(encoding="utf-8")
    normalized = " ".join(text.split())

    assert (
        "restart every other long-running process that imports Hermes or LCM-X"
        in normalized
    )
    assert "after `hermes update`" in normalized


@pytest.mark.parametrize("doc_path", DOCS)
def test_troubleshooting_maps_stale_process_symptoms_to_restart(doc_path):
    text = (REPO_ROOT / doc_path).read_text(encoding="utf-8")
    normalized = " ".join(text.split())

    assert (
        "### Errors in one process after `hermes update` or an LCM-X update"
        in text
    )
    assert "ImportError" in normalized
    assert "unexpected keyword argument" in normalized
    assert "Restart that process" in normalized


@pytest.mark.parametrize("doc_path", DOCS)
def test_update_section_links_to_the_troubleshooting_anchor(doc_path):
    text = (REPO_ROOT / doc_path).read_text(encoding="utf-8")

    assert "(#errors-in-one-process-after-hermes-update-or-an-lcm-x-update)" in text
