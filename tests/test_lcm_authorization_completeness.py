from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from pathlib import Path

from hermes_lcm.engine import LCM_TOOL_TARGET_BINDINGS


REPO_ROOT = Path(__file__).resolve().parents[1]
INVENTORY_PATH = REPO_ROOT / "access_context" / "inventory.json"


@dataclass(frozen=True)
class HookCall:
    site: str
    line: int
    node: ast.Call
    function: ast.AST | None


class _HookVisitor(ast.NodeVisitor):
    def __init__(self, module: str) -> None:
        self.module = module
        self.stack: list[str] = []
        self.functions: list[ast.AST] = []
        self.calls: list[HookCall] = []

    def _site(self) -> str:
        suffix = ".".join(self.stack)
        return f"{self.module}:{suffix}" if suffix else self.module

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        self.stack.append(node.name)
        self.functions.append(node)
        self.generic_visit(node)
        self.functions.pop()
        self.stack.pop()

    def visit_Call(self, node: ast.Call) -> None:
        if (
            isinstance(node.func, ast.Name)
            and node.func.id == "policy_for_engine"
        ):
            self.calls.append(
                HookCall(self._site(), node.lineno, node, self.functions[-1] if self.functions else None)
            )
        self.generic_visit(node)


def _inventory_payload() -> list[dict[str, object]]:
    raw = json.loads(INVENTORY_PATH.read_text(encoding="utf-8"))
    assert isinstance(raw, list) and raw
    assert all(isinstance(entry, dict) for entry in raw)
    return raw


def _source_files() -> tuple[Path, ...]:
    excluded = {"tests", "bench", "benchmarks", "__pycache__"}
    return tuple(
        sorted(
            path
            for path in REPO_ROOT.rglob("*.py")
            if not any(part.startswith(".venv") or part in excluded for part in path.parts)
        )
    )


def _hook_calls() -> tuple[HookCall, ...]:
    calls: list[HookCall] = []
    for path in _source_files():
        relative = path.relative_to(REPO_ROOT).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=relative)
        visitor = _HookVisitor(relative)
        visitor.visit(tree)
        calls.extend(visitor.calls)
    return tuple(calls)


def _hook_specs(entries: list[dict[str, object]]) -> tuple[dict[str, object], ...]:
    specs: list[dict[str, object]] = []
    for entry in entries:
        entry_id = str(entry.get("id"))
        hook = entry.get("hook")
        assert isinstance(hook, dict), f"{entry_id} must declare structured hook metadata"
        required = hook.get("required")
        assert isinstance(required, bool), f"{entry_id} hook.required must be boolean"
        if required:
            assert entry.get("authority_requirement") != "none_required", (
                f"{entry_id} requires a hook but has none_required authority"
            )
            sites = hook.get("sites")
            assert isinstance(sites, list) and sites and all(
                isinstance(site, str) and site.strip() for site in sites
            ), f"{entry_id} hook sites must be non-empty strings"
            assert not hook.get("reason"), f"{entry_id} hooked entries cannot carry a hook-free reason"
        else:
            assert entry.get("authority_requirement") == "none_required", (
                f"{entry_id} hook-free entries must use none_required authority"
            )
            reason = hook.get("reason")
            assert isinstance(reason, str) and reason.strip(), (
                f"{entry_id} hook-free entries require a non-empty reason"
            )
            assert hook.get("sites", []) in ([], None), f"{entry_id} hook-free entries cannot list sites"
        specs.append({"id": entry_id, "required": required, **hook})
    return tuple(specs)


def test_inventory_binds_every_required_entry_to_structural_hooks() -> None:
    entries = _inventory_payload()
    specs = _hook_specs(entries)
    calls = _hook_calls()
    observed_sites = {call.site for call in calls}
    claimed_sites: dict[str, list[str]] = {}

    for spec in specs:
        if not spec["required"]:
            continue
        for site in spec["sites"]:
            assert site in observed_sites, f"{spec['id']} requires missing hook {site}"
            claimed_sites.setdefault(site, []).append(str(spec["id"]))

    unclassified = sorted(observed_sites - set(claimed_sites))
    assert not unclassified, f"hook sites missing inventory entries: {unclassified}"


def test_hook_sites_resolve_only_through_access_policy_seam() -> None:
    calls = _hook_calls()
    assert calls
    calls_by_module: dict[str, list[HookCall]] = {}
    for call in calls:
        calls_by_module.setdefault(call.site.split(":", 1)[0], []).append(call)

    for module, module_calls in calls_by_module.items():
        path = REPO_ROOT / module
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=module)
        access_policy_imports = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "_access_policy" for target in node.targets)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Attribute)
            and isinstance(node.value.func.value, ast.Name)
            and node.value.func.value.id == "importlib"
            and node.value.func.attr == "import_module"
            and len(node.value.args) == 1
            and isinstance(node.value.args[0], ast.Constant)
            and node.value.args[0].value == "access_policy"
        ]
        assert access_policy_imports, f"{module} does not import access_policy for its seam"
        seam_bindings = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "policy_for_engine" for target in node.targets)
            and isinstance(node.value, ast.Attribute)
            and node.value.attr == "policy_for_engine"
            and isinstance(node.value.value, ast.Name)
            and node.value.value.id == "_access_policy"
        ]
        assert seam_bindings, f"{module} does not bind policy_for_engine from access_policy"

        for call in module_calls:
            assert isinstance(call.node.func, ast.Name)
            assert call.node.func.id == "policy_for_engine"
            if call.function is None:
                continue
            for node in ast.walk(call.function):
                if isinstance(node, ast.Call):
                    assert not (
                        isinstance(node.func, ast.Name)
                        and node.func.id in {"resolve_policy", "TrustedOwnerPolicy", "FailClosedPolicy"}
                    ), f"{call.site} resolves policy outside policy_for_engine"
                    if (
                        isinstance(node.func, ast.Name)
                        and node.func.id == "getattr"
                        and len(node.args) >= 2
                        and isinstance(node.args[1], ast.Constant)
                        and node.args[1].value in {"lcm_teams_enabled", "get_lcm_access_context"}
                    ):
                        raise AssertionError(f"{call.site} reads policy wiring directly")
                if isinstance(node, ast.Attribute) and node.attr in {
                    "lcm_teams_enabled",
                    "get_lcm_access_context",
                    "resolve_policy",
                    "TrustedOwnerPolicy",
                    "FailClosedPolicy",
                }:
                    raise AssertionError(f"{call.site} reads policy wiring directly")


def test_tool_authority_paths_are_discovered_from_source() -> None:
    tree = ast.parse((REPO_ROOT / "tools.py").read_text(encoding="utf-8"), filename="tools.py")
    source_tools = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("lcm_")
    }
    inventory_tools = {
        str(entry["entry_point"])
        for entry in _inventory_payload()
        if entry.get("module") == "tools.py" and str(entry.get("entry_point", "")).startswith("lcm_")
    }
    assert source_tools == inventory_tools, (
        f"source lcm_* handlers differ from inventory: "
        f"source-only={sorted(source_tools - inventory_tools)}, "
        f"inventory-only={sorted(inventory_tools - source_tools)}"
    )


def test_tool_target_bindings_cover_source_and_inventory_both_directions() -> None:
    tree = ast.parse((REPO_ROOT / "tools.py").read_text(encoding="utf-8"), filename="tools.py")
    source_tools = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("lcm_")
    }
    mapping_tools = set(LCM_TOOL_TARGET_BINDINGS)
    assert mapping_tools == source_tools, (
        "target binding map differs from source lcm_* handlers: "
        f"source-only={sorted(source_tools - mapping_tools)}, "
        f"mapping-only={sorted(mapping_tools - source_tools)}"
    )

    inventory_entries = {
        str(entry["entry_point"]): entry
        for entry in _inventory_payload()
        if entry.get("module") == "tools.py" and str(entry.get("entry_point", "")).startswith("lcm_")
    }
    assert set(inventory_entries) == mapping_tools
    for tool_name in sorted(source_tools):
        entry = inventory_entries[tool_name]
        binding = entry.get("target_binding")
        assert isinstance(binding, dict), f"{tool_name} must declare target_binding"
        expected = LCM_TOOL_TARGET_BINDINGS[tool_name]
        expected_args = list(expected.get("args", ()))
        assert binding.get("args") == expected_args, f"{tool_name} target args drifted"
        target_free = bool(expected.get("target_free", False))
        assert bool(binding.get("target_free", False)) is target_free
        if target_free:
            reason = binding.get("reason")
            assert isinstance(reason, str) and reason.strip(), (
                f"{tool_name} target_free entries require a non-empty reason"
            )
        else:
            assert not binding.get("reason"), f"{tool_name} target-bound entries cannot carry a target-free reason"
