"""Schema and loader for the machine-checkable authority-path inventory."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


CATEGORIES = frozenset(
    {"ingress", "sessions", "cron", "callbacks", "retrieval", "expansion", "writes", "maintenance", "admin"}
)
AUTHORITY_REQUIREMENTS = frozenset(
    {"read_scoped", "write_scoped", "owner_only", "admin_only", "none_required"}
)
DISCLOSURES = frozenset({"existence", "count", "ranking", "content", "handle"})


class InventoryError(ValueError):
    """Raised for malformed or empty authority-path inventory data."""


@dataclass(frozen=True)
class InventoryHook:
    """The wiring obligation a later slice owes this authority path.

    This is the machine-checkable half of the inventory: the completeness test
    grades the wiring slices against ``sites``, so dropping it would leave the
    obligation as prose only.
    """

    required: bool
    sites: tuple[str, ...] = ()
    reason: str = ""

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "InventoryHook":
        if "required" not in payload:
            raise InventoryError("inventory hook must declare 'required'")
        if not isinstance(payload["required"], bool):
            raise InventoryError("inventory hook 'required' must be a boolean")
        hook = cls(
            required=payload["required"],
            sites=tuple(str(item) for item in payload.get("sites", ())),
            reason=str(payload.get("reason", "")),
        )
        hook.validate()
        return hook

    def validate(self) -> None:
        if self.required:
            if not self.sites:
                raise InventoryError("a required hook must name at least one site")
            for site in self.sites:
                module, separator, symbol = site.partition(":")
                if not separator or not module.endswith(".py") or not symbol:
                    raise InventoryError(f"malformed hook site: {site}")
        else:
            if self.sites:
                raise InventoryError("an exempt hook must not name sites")
            if not self.reason:
                raise InventoryError("an exempt hook must record why no hook is needed")


@dataclass(frozen=True)
class InventoryEntry:
    id: str
    category: str
    module: str
    entry_point: str
    authority_requirement: str
    discloses: tuple[str, ...]
    notes: str
    hook: InventoryHook

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "InventoryEntry":
        required = (
            "id",
            "category",
            "module",
            "entry_point",
            "authority_requirement",
            "discloses",
            "notes",
            "hook",
        )
        missing = [name for name in required if name not in payload]
        if missing:
            raise InventoryError(f"inventory entry missing fields: {', '.join(missing)}")
        if not isinstance(payload["hook"], Mapping):
            raise InventoryError(f"inventory hook must be an object: {payload['id']}")
        entry = cls(
            id=str(payload["id"]),
            category=str(payload["category"]),
            module=str(payload["module"]),
            entry_point=str(payload["entry_point"]),
            authority_requirement=str(payload["authority_requirement"]),
            discloses=tuple(str(item) for item in payload["discloses"]),
            notes=str(payload["notes"]),
            hook=InventoryHook.from_mapping(payload["hook"]),
        )
        entry.validate()
        return entry

    def validate(self) -> None:
        if not self.id or not self.module or not self.entry_point:
            raise InventoryError("inventory identifiers and paths must be non-empty")
        if self.category not in CATEGORIES:
            raise InventoryError(f"unknown inventory category: {self.category}")
        if self.authority_requirement not in AUTHORITY_REQUIREMENTS:
            raise InventoryError(f"unknown authority requirement: {self.authority_requirement}")
        if any(item not in DISCLOSURES for item in self.discloses):
            raise InventoryError(f"unknown disclosure in {self.id}")
        if not self.notes:
            raise InventoryError(f"inventory notes must be non-empty: {self.id}")


def inventory_path() -> Path:
    return Path(__file__).resolve().with_name("inventory.json")


def load_inventory(path: str | Path | None = None) -> tuple[InventoryEntry, ...]:
    """Load non-empty inventory JSON from a source checkout or installation."""

    candidate = Path(path) if path is not None else inventory_path()
    if not candidate.is_file():
        raise FileNotFoundError(f"authority-path inventory not found: {candidate}")
    raw = json.loads(candidate.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not raw:
        raise InventoryError("authority-path inventory must be a non-empty JSON array")
    entries = tuple(InventoryEntry.from_mapping(item) for item in raw)
    ids = [entry.id for entry in entries]
    if len(ids) != len(set(ids)):
        raise InventoryError("authority-path inventory IDs must be unique")
    return entries


def parse_manifest_tools(manifest_path: str | Path) -> frozenset[str]:
    """Parse ``provides_tools`` textually, avoiding a YAML runtime dependency."""

    path = Path(manifest_path)
    in_tools = False
    tools: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped == "provides_tools:":
            in_tools = True
            continue
        if in_tools and stripped.startswith("-"):
            name = stripped[1:].strip()
            if name:
                tools.add(name)
            continue
        if in_tools and stripped and not line.startswith((" ", "\t")):
            break
    return frozenset(tools)


def validate_inventory_against_repo(
    entries: Iterable[InventoryEntry], *, repo_root: str | Path, manifest_path: str | Path | None = None
) -> None:
    """Apply the machine-checkable inventory invariants used by conformance tests."""

    items = tuple(entries)
    if not items:
        raise InventoryError("inventory is empty")
    root = Path(repo_root)
    missing = [entry.module for entry in items if not (root / entry.module).is_file()]
    if missing:
        raise InventoryError(f"inventory references missing modules: {', '.join(missing)}")
    missing_sites = [
        site
        for entry in items
        for site in entry.hook.sites
        if not (root / site.partition(":")[0]).is_file()
    ]
    if missing_sites:
        raise InventoryError(f"inventory hook sites reference missing modules: {', '.join(missing_sites)}")
    categories = {entry.category for entry in items}
    if categories != CATEGORIES:
        raise InventoryError(f"inventory categories differ: missing={sorted(CATEGORIES - categories)}")
    manifest = Path(manifest_path) if manifest_path is not None else root / "plugin.yaml"
    manifest_tools = parse_manifest_tools(manifest)
    inventory_tools = {entry.entry_point for entry in items if entry.entry_point.startswith("lcm_")}
    if inventory_tools != manifest_tools:
        raise InventoryError(
            f"tool coverage differs: inventory-only={sorted(inventory_tools - manifest_tools)}, "
            f"manifest-only={sorted(manifest_tools - inventory_tools)}"
        )
