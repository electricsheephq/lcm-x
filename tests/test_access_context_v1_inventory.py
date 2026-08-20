from __future__ import annotations

import re
from pathlib import Path

import pytest

from hermes_lcm.access_context.inventory import (
    AUTHORITY_REQUIREMENTS,
    CATEGORIES,
    DISCLOSURES,
    InventoryEntry,
    InventoryError,
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


def test_inventory_covers_non_tool_authority_paths() -> None:
    entries = load_inventory()
    non_tool = [entry for entry in entries if entry.module != "tools.py"]
    assert len({entry.module for entry in non_tool}) > 1
    for category in ("writes", "maintenance", "callbacks", "sessions"):
        assert any(entry.category == category for entry in non_tool)


def _entry_payload(**overrides: object) -> dict:
    payload = {
        "id": "ingress-example",
        "category": "ingress",
        "module": "tools.py",
        "entry_point": "lcm_grep",
        "authority_requirement": "read_scoped",
        "discloses": ["existence"],
        "notes": "example",
        "hook": {"required": True, "sites": ["engine.py:LCMEngine.handle_tool_call"]},
    }
    payload.update(overrides)
    return payload


def test_inventory_entries_retain_hook_obligations() -> None:
    # The hook obligation is the machine-checkable half of this inventory: the
    # wiring slices are graded against it, so the loader must model it rather
    # than drop it on the floor.
    entries = load_inventory()
    required = [entry for entry in entries if entry.hook.required]
    exempt = [entry for entry in entries if not entry.hook.required]
    assert len(required) >= 15
    assert all(entry.hook.sites for entry in required)
    assert all(":" in site for entry in required for site in entry.hook.sites)
    assert exempt and all(entry.hook.reason for entry in exempt)
    assert all(not entry.hook.sites for entry in exempt)


def test_malformed_hook_metadata_is_rejected_not_dropped() -> None:
    InventoryEntry.from_mapping(_entry_payload())
    with pytest.raises(InventoryError):
        InventoryEntry.from_mapping({k: v for k, v in _entry_payload().items() if k != "hook"})
    with pytest.raises(InventoryError):
        InventoryEntry.from_mapping(_entry_payload(hook={"required": True, "sites": []}))
    with pytest.raises(InventoryError):
        InventoryEntry.from_mapping(_entry_payload(hook={"required": True, "sites": ["engine.py"]}))
    with pytest.raises(InventoryError):
        InventoryEntry.from_mapping(_entry_payload(hook={"required": False, "reason": ""}))
    with pytest.raises(InventoryError):
        InventoryEntry.from_mapping(
            _entry_payload(hook={"required": False, "reason": "why", "sites": ["engine.py:x"]})
        )


def test_hook_sites_reference_modules_that_exist() -> None:
    entries = load_inventory()
    validate_inventory_against_repo(entries, repo_root=REPO_ROOT)
    forged = InventoryEntry.from_mapping(
        _entry_payload(hook={"required": True, "sites": ["no_such_module.py:hook"]})
    )
    with pytest.raises(InventoryError):
        validate_inventory_against_repo((*entries, forged), repo_root=REPO_ROOT)


def test_non_tool_entry_points_exist_in_their_modules() -> None:
    entries = load_inventory()
    for entry in entries:
        if entry.module == "tools.py":
            continue
        source = (REPO_ROOT / entry.module).read_text(encoding="utf-8")
        symbol = re.escape(entry.entry_point)
        assert any(
            re.search(pattern, source, re.MULTILINE)
            for pattern in (
                rf"\bdef\s+{symbol}\b",
                rf"\bclass\s+{symbol}\b",
                rf"^{symbol}\s*=",
            )
        ), f"{entry.module} does not define {entry.entry_point}"
