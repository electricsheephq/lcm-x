from __future__ import annotations

from pathlib import Path

from access_context.inventory import (
    AUTHORITY_REQUIREMENTS,
    CATEGORIES,
    DISCLOSURES,
    load_inventory,
    parse_manifest_tools,
    validate_inventory_against_repo,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_inventory_is_non_empty_and_machine_checkable() -> None:
    entries = load_inventory()
    assert len(entries) >= 15
    validate_inventory_against_repo(entries, repo_root=REPO_ROOT)


def test_tool_coverage_is_exact_both_ways() -> None:
    entries = load_inventory()
    manifest_tools = parse_manifest_tools(REPO_ROOT / "plugin.yaml")
    inventory_tools = {entry.entry_point for entry in entries if entry.entry_point.startswith("lcm_")}
    assert inventory_tools == manifest_tools
    assert len(inventory_tools) == 15


def test_inventory_categories_paths_and_closed_vocabularies() -> None:
    entries = load_inventory()
    assert {entry.category for entry in entries} == CATEGORIES
    assert all((REPO_ROOT / entry.module).is_file() for entry in entries)
    assert all(entry.authority_requirement in AUTHORITY_REQUIREMENTS for entry in entries)
    assert all(set(entry.discloses) <= DISCLOSURES for entry in entries)
    assert all(entry.notes for entry in entries)
