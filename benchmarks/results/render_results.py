"""Render the public results snapshot using only the standard library."""

import json
from pathlib import Path


DATA_PATH = Path(__file__).with_name("public-results.json")
PAGE_PATH = DATA_PATH.parents[1] / "RESULTS.md"


def placement(data, benchmark):
    """Competition rank among backends with a score; missing cells do not count."""
    scores = {
        row["backend"]: row["scores"][benchmark]
        for row in data["snapshot"]["rows"]
        if row["scores"][benchmark] is not None
    }
    value = scores["LCM-X"]
    rank = 1 + sum(score > value for score in scores.values())
    suffix = "th" if 10 <= rank % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(rank % 10, "th")
    return f"{rank}{suffix} of {len(scores)}"


def reported_entries(data):
    return [entry for entry in data["entries"] if entry["status"] == "reported"]


def strongest_entries(data):
    return sorted(
        reported_entries(data),
        key=lambda entry: int(placement(data, entry["benchmark"]).split()[0][:-2]),
    )[:2]


SHARED_FIELDS = (
    "run_date", "lcmx_version", "executable_identity", "config", "harness_commit",
    "claim_class", "source_url", "evidence_url",
)


def shared_metadata(entries):
    """Page-level metadata is rendered once, so every reported entry must agree on it."""
    first = entries[0]
    for entry in entries[1:]:
        for field in SHARED_FIELDS:
            if entry[field] != first[field]:
                raise ValueError(f"{entry['benchmark']} differs from {first['benchmark']} on {field}")
    return first


def render_readme_results(data):
    """The short README subsection shares the page's data and ranking logic."""
    entries = strongest_entries(data)
    placements = "; ".join(
        f'[{entry["benchmark"]} {entry["value"]:.2f}: {placement(data, entry["benchmark"])} scored backends]({entry["source_url"]}#result-summary)'
        for entry in entries
    )
    first = shared_metadata(reported_entries(data))
    return (
        "### Results\n\n"
        f'[{data["board"]}]({first["source_url"]}): {first["claim_class"]}.\n'
        f"{placements}.\n"
        "[Results, configuration, evidence and disclosures](benchmarks/RESULTS.md).\n"
    )


def render_results(data):
    entries = reported_entries(data)
    first = shared_metadata(entries)
    not_run = [entry["benchmark"] for entry in data["entries"] if entry["status"] == "not_run"]
    # A submitted entry carries its own run metadata; it joins the table only once upstream accepts it.
    submitted = [entry for entry in data["entries"] if entry["status"] == "submitted"]
    lines = [
        "# Public results",
        "",
        f'{data["board"]}: {data["description"]}',
        "The LCM-X maintainers ran these rows on the unchanged upstream pipeline and prompts.",
        f'Claim class: **{first["claim_class"]}**.',
        "Placements count only listed backends with a score in the pinned upstream summary; higher scores rank first.",
        "The upstream page does not claim an optimal configuration for every backend.",
        "",
        "| Benchmark | Metric | LCM-X value | Placement | Links |",
        "| --- | --- | ---: | --- | --- |",
    ]
    for entry in entries:
        lines.append(
            f'| {entry["benchmark"]} | {entry["metric"]} | {entry["value"]:.2f} | '
            f'{placement(data, entry["benchmark"])} | '
            f'[Source]({entry["source_url"]}#result-summary) · [Evidence]({entry["evidence_url"]}) |'
        )
    lines += [
        "",
        f'Run date: {first["run_date"]}. LCM-X {first["lcmx_version"]}; {first["executable_identity"]}.',
        f'Configuration: {first["config"]["embeddings"]} embeddings; top-k {first["config"]["top_k"]}.',
        f'Harness: [`{first["harness_commit"]}`](https://github.com/electricsheephq/OmniMemEval/tree/{first["harness_commit"]}).',
        "",
        "## Disclosures",
        "",
        *dict.fromkeys(entry["disclosures"] for entry in entries + submitted),
        "",
        *(["Not yet run: " + ", ".join(not_run) + "."] if not_run else []),
        *(
            f'Submitted upstream, pending review: {entry["benchmark"]} {entry["metric"]} {entry["value"]:.2f} '
            f'([upstream review]({entry["source_url"]}) · [Evidence]({entry["evidence_url"]})); '
            f'{entry["run_date"]} run, LCM-X {entry["lcmx_version"]}, harness `{entry["harness_commit"][:8]}`.'
            for entry in submitted
        ),
        f'In review: our {data["in_review"]["board"]} entry; the board publishes its own results.',
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    PAGE_PATH.write_text(render_results(json.loads(DATA_PATH.read_text(encoding="utf-8"))), encoding="utf-8")
