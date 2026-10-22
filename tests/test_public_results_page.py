"""Keep public claims tied to the pinned, offline upstream summary snapshot."""

import importlib.util
import json
import re
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "benchmarks/results/public-results.json"
RENDER_PATH = DATA_PATH.with_name("render_results.py")
PAGE_PATH = ROOT / "benchmarks/RESULTS.md"
spec = importlib.util.spec_from_file_location("public_results_renderer", RENDER_PATH)
renderer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(renderer)


@pytest.fixture
def data():
    return json.loads(DATA_PATH.read_text(encoding="utf-8"))


def ranked_scores(data, benchmark):
    return sorted(
        (row["scores"][benchmark] for row in data["snapshot"]["rows"]
         if row["scores"][benchmark] is not None),
        reverse=True,
    )


def test_page_is_fresh(data):
    assert PAGE_PATH.read_text(encoding="utf-8") == renderer.render_results(data)


def test_entries_have_provenance_and_match_snapshot(data):
    snapshot = data["snapshot"]
    lcmx = next(row["scores"] for row in snapshot["rows"] if row["backend"] == "LCM-X")
    assert {entry["benchmark"] for entry in data["entries"]} == set(snapshot["benchmarks"])
    assert sum(entry["status"] == "reported" for entry in data["entries"]) == 5
    for entry in data["entries"]:
        for field in (
            "board", "benchmark", "metric", "lcmx_version", "executable_identity",
            "config", "run_date", "harness_commit", "source_url", "evidence_url",
            "claim_class", "disclosures",
        ):
            assert entry[field], (entry["benchmark"], field)
        assert entry["evidence_url"].startswith("https://github.com/")
        assert (entry["status"] == "not_run") == (entry["value"] is None)
        assert entry["config"]["embeddings"] and entry["config"]["top_k"] > 0
        if entry["status"] == "submitted":
            # Pending upstream review: never in the pinned snapshot, never ranked.
            assert entry["source_url"].startswith("https://github.com/MemTensor/OmniMemEval/pull/")
            assert entry["claim_class"] == "contributor-run, submitted upstream"
            assert lcmx[entry["benchmark"]] is None and entry["value"] is not None
            continue
        assert entry["source_url"] == snapshot["source_url"]
        assert "/blob/f7476c55/" in entry["source_url"]
        assert entry["claim_class"] == "contributor-run, accepted upstream"
        assert entry["value"] == lcmx[entry["benchmark"]]
    assert [entry["benchmark"] for entry in data["entries"] if entry["status"] == "not_run"] == []
    assert [(entry["benchmark"], entry["value"]) for entry in data["entries"]
            if entry["status"] == "submitted"] == [("BEAM 10M", 63.32)]


def test_page_values_and_placements_match_snapshot(data):
    page = PAGE_PATH.read_text(encoding="utf-8")
    rows = [line.split("|")[1:-1] for line in page.splitlines() if line.startswith("| ")][2:]
    entries = [entry for entry in data["entries"] if entry["status"] == "reported"]
    assert len(rows) == len(entries)
    for row, entry in zip(rows, entries, strict=True):
        benchmark, metric, value, place, links = (cell.strip() for cell in row)
        assert benchmark == entry["benchmark"]
        assert metric == entry["metric"]
        assert value == f'{entry["value"]:.2f}'
        match = re.fullmatch(r"(\d+)(st|nd|rd|th) of (\d+)", place)
        assert match, place
        rank, suffix, count = match.groups()
        scores = ranked_scores(data, benchmark)
        expected_rank = scores.index(entry["value"]) + 1
        assert int(rank) == expected_rank
        assert int(count) == len(scores)
        expected_suffix = "th" if 10 <= expected_rank % 100 <= 20 else {
            1: "st", 2: "nd", 3: "rd",
        }.get(expected_rank % 10, "th")
        assert suffix == expected_suffix
        assert entry["source_url"] in links and entry["evidence_url"] in links
    assert "Not yet run" not in page
    submitted = next(entry for entry in data["entries"] if entry["status"] == "submitted")
    line = next(line for line in page.splitlines() if line.startswith("Submitted upstream, pending review: "))
    assert "BEAM 10M nugget score 63.32 " in line and submitted["source_url"] in line
    assert submitted["evidence_url"] in line and submitted["disclosures"] in page
    # A resumed run names every harness commit with its role, ending at the one that scored it.
    assert [c["sha"][:8] for c in submitted["harness_commits"]] == ["cd955c46", "282be0b8"]
    assert submitted["harness_commits"][-1]["sha"] == submitted["harness_commit"]
    for commit in submitted["harness_commits"]:
        assert f'`{commit["sha"][:8]}` ({commit["role"]})' in line
    assert "| BEAM 10M |" not in page
    expected_review = "our Agent Memory Leaderboard Cycle 2 entry; the board publishes its own results."
    assert f"In review: {expected_review}" in page
    assert page.split("In review:")[1].strip() == expected_review
    for entry in entries:
        assert entry["disclosures"] in page


def test_readme_results_only_quote_sourced_claims(data):
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    status = readme.split("## Project status\n", 1)[1].split("\n## ", 1)[0]
    subsection = "### Results\n" + status.split("### Results\n", 1)[1]
    assert subsection.rstrip() == renderer.render_readme_results(data).rstrip()
    assert 3 <= len(subsection.strip().splitlines()) <= 6
    lines = subsection.strip().splitlines()
    first = data["entries"][0]
    assert lines[2] == f'[{data["board"]}]({first["source_url"]}): {first["claim_class"]}.'
    entries = [entry for entry in data["entries"] if entry["status"] == "reported"]
    strongest = sorted(entries, key=lambda entry: ranked_scores(data, entry["benchmark"]).index(entry["value"]))[:2]
    claims = []
    for entry in strongest:
        scores = ranked_scores(data, entry["benchmark"])
        rank = scores.index(entry["value"]) + 1
        suffix = {1: "st", 2: "nd"}[rank]
        claims.append(
            f'[{entry["benchmark"]} {entry["value"]:.2f}: {rank}{suffix} of {len(scores)} scored backends]({entry["source_url"]}#result-summary)'
        )
    assert lines[3] == "; ".join(claims) + "."
    assert lines[4] == "[Results, configuration, evidence and disclosures](benchmarks/RESULTS.md)."


def test_divergent_entry_metadata_is_refused(data):
    import copy
    changed = copy.deepcopy(data)
    reported = [entry for entry in changed["entries"] if entry["status"] == "reported"]
    reported[1]["run_date"] = "2099-01-01"
    with pytest.raises(ValueError, match="run_date"):
        renderer.render_results(changed)
    with pytest.raises(ValueError, match="run_date"):
        renderer.render_readme_results(changed)


def test_not_run_line_renders_only_when_an_entry_is_not_run(data):
    import copy
    changed = copy.deepcopy(data)
    entry = next(entry for entry in changed["entries"] if entry["status"] == "submitted")
    entry.update(status="not_run", value=None)
    page = renderer.render_results(changed)
    assert "Not yet run: BEAM 10M." in page
    assert "Submitted upstream, pending review" not in page


def highlight_block():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    return re.search(
        r"<!-- results-highlights:start -->.*?<!-- results-highlights:end -->",
        readme, re.DOTALL,
    ).group()


def table_rows(text):
    return [line for line in text.splitlines() if line.startswith("| ")]


def test_highlights_are_fresh(data):
    assert highlight_block() == renderer.render_highlights(data)


def test_highlights_follow_tagline_before_contents():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    tagline = "> Bounded context, unbounded memory. Nothing is ever lost."
    block = highlight_block()
    assert readme.index(tagline) < readme.index(block) < readme.index("LCM-X replaces")
    assert readme.index(block) < readme.index("## Table of contents")
    assert readme.split(tagline, 1)[1].startswith("\n\n" + block + "\n\nLCM-X replaces")


def test_highlights_order_by_placement_with_stable_ties(data):
    rows = table_rows(renderer.render_highlights(data))[2:]
    entries = renderer.reported_entries(data)
    expected = sorted(
        entries,
        key=lambda entry: ranked_scores(data, entry["benchmark"]).index(entry["value"]),
    )
    assert [row.split("|")[1].strip() for row in rows[:len(entries)]] == [
        entry["benchmark"] for entry in expected
    ]
    for row, entry in zip(rows, expected):
        assert f'| {entry["value"]:.2f} |' in row
        assert f'[{renderer.placement(data, entry["benchmark"])}]({entry["source_url"]}#result-summary)' in row


def test_plugin_description_placements_match_strongest_entries(data):
    line = next(line for line in (ROOT / "plugin.yaml").read_text().splitlines()
                if line.startswith("description:"))
    expected = ", ".join(
        f'{entry["benchmark"]} {renderer.placement(data, entry["benchmark"])}'
        for entry in renderer.strongest_entries(data)
    )
    assert json.loads(line.partition(":")[2].strip()) == (
        "LCM-X — Lossless Context Memory eXtension for Hermes, with a recoverable DAG-based context engine"
        f" OmniMemEval: {expected}."
    )


def test_release_notes_match_readme_table():
    import subprocess
    import sys
    output = subprocess.run(
        [sys.executable, str(RENDER_PATH), "--release-notes"],
        check=True, capture_output=True, text=True,
    ).stdout
    assert "<!--" not in output
    assert table_rows(output) == table_rows(highlight_block())
    assert output.strip() == highlight_block().split("\n", 1)[1].rsplit("\n", 1)[0]


def test_submitted_highlight_has_review_link_without_placement(data):
    submitted = next(entry for entry in data["entries"] if entry["status"] == "submitted")
    row = next(row for row in table_rows(renderer.render_highlights(data))
               if row.startswith(f'| {submitted["benchmark"]} |'))
    assert row == (
        f'| {submitted["benchmark"]} | {submitted["value"]:.2f} | '
        f'[in upstream review]({submitted["source_url"]}) |'
    )
    assert not re.search(r"\d+(?:st|nd|rd|th) of", row)
