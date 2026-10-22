"""Render the public results snapshot using only the standard library."""

import argparse
import json
import re
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
    """Keep numbers in one generated README table; this subsection points to it."""
    shared_metadata(reported_entries(data))
    return (
        "### Results\n\n"
        "See the [benchmark highlights](#benchmark-highlights) above and "
        "[results, configuration, evidence and disclosures](benchmarks/RESULTS.md).\n"
    )


def render_highlights(data, *, release_notes=False):
    """Render the README and release-note showcase from the public snapshot."""
    first = shared_metadata(reported_entries(data))
    entries = sorted(
        reported_entries(data),
        key=lambda entry: int(placement(data, entry["benchmark"]).split()[0][:-2]),
    )
    lines = [
        "<!-- results-highlights:start -->",
        f'<a id="benchmark-highlights"></a>\n**Benchmark highlights** ([OmniMemEval]({first["source_url"]}), {first["claim_class"]})',
        "",
        "| Benchmark | LCM-X | Placement |",
        "| --- | ---: | --- |",
    ]
    for entry in entries:
        lines.append(
            f'| {entry["benchmark"]} | {entry["value"]:.2f} | '
            f'[{placement(data, entry["benchmark"])}]({entry["source_url"]}#result-summary) |'
        )
    for entry in data["entries"]:
        if entry["status"] == "submitted":
            lines.append(
                f'| {entry["benchmark"]} | {entry["value"]:.2f} | '
                f'[in upstream review]({entry["source_url"]}) |'
            )
    results_url = (
        "https://github.com/electricsheephq/lcm-x/blob/main/benchmarks/RESULTS.md"
        if release_notes else "benchmarks/RESULTS.md"
    )
    lines += [
        "",
        f"Configuration, evidence and disclosures: [benchmarks/RESULTS.md]({results_url}).",
        "<!-- results-highlights:end -->",
    ]
    return "\n".join(lines)


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
            f'{entry["run_date"]} run, LCM-X {entry["lcmx_version"]}, harness '
            + " and ".join(f'`{c["sha"][:8]}` ({c["role"]})'
                           for c in entry.get("harness_commits") or [{"sha": entry["harness_commit"], "role": "all stages"}])
            + "."

            for entry in submitted
        ),
        f'In review: our {data["in_review"]["board"]} entry; the board publishes its own results.',
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-notes", action="store_true")
    args = parser.parse_args()
    data = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    block = render_highlights(data, release_notes=args.release_notes)
    if args.release_notes:
        print("\n".join(block.splitlines()[1:-1]))
    else:
        PAGE_PATH.write_text(render_results(data), encoding="utf-8")
        root = DATA_PATH.parents[2]
        readme_path = root / "README.md"
        readme = readme_path.read_text(encoding="utf-8")
        if "<!-- results-highlights:start -->" in readme:
            readme = re.sub(
                r"<!-- results-highlights:start -->.*?<!-- results-highlights:end -->",
                lambda match: block, readme, flags=re.DOTALL,
            )
        else:
            tagline = "> Bounded context, unbounded memory. Nothing is ever lost."
            readme = readme.replace(tagline + "\n", tagline + "\n\n" + block + "\n", 1)
        readme = re.sub(
            r"### Results\n.*?(?=\n## |\Z)",
            lambda match: render_readme_results(data), readme, count=1, flags=re.DOTALL,
        )
        readme_path.write_text(readme, encoding="utf-8")
        plugin_path = root / "plugin.yaml"
        plugin = plugin_path.read_text(encoding="utf-8")
        placements = ", ".join(
            f'{entry["benchmark"]} {placement(data, entry["benchmark"])}'
            for entry in strongest_entries(data)
        )
        def description(match):
            sentence = json.loads(match[1]).split(" OmniMemEval:", 1)[0]
            return "description: " + json.dumps(
                f"{sentence} OmniMemEval: {placements}.", ensure_ascii=False,
            )
        plugin = re.sub(r'^description: (".*")$', description, plugin, flags=re.MULTILINE)
        plugin_path.write_text(plugin, encoding="utf-8")
