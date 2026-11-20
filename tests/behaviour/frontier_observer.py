"""Observe forwarded and holder-only frontier state."""
from __future__ import annotations

import ast
import dataclasses
import importlib
from typing import Any

engine_module: Any = importlib.import_module("hermes_lcm.engine")
HOLDER_NAME = "_frontier"
REFUSED_INPUT = "_frontier.refused_input"
holder_type: Any = getattr(engine_module, "FrontierState", None)
LEGACY_FIELDS: dict[str, str] = {
    field.name: name
    for field in dataclasses.fields(holder_type)
    for name, prop in vars(engine_module.LCMEngine).items()
    if isinstance(prop, property) and prop.fget is not None
    and (HOLDER_NAME in prop.fget.__code__.co_names or "_frontier_holder" in prop.fget.__code__.co_names)
    and field.name in prop.fget.__code__.co_names
} if holder_type is not None else {}


def frontier_fields(method: ast.AST) -> set[str]:
    """Translate only the holder fields actually accessed by this method."""
    names = LEGACY_FIELDS | {"refused_input": REFUSED_INPUT}
    return {
        names[node.attr]
        for node in ast.walk(method)
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Attribute)
        and node.value.attr == HOLDER_NAME
        and isinstance(node.value.value, ast.Name) and node.value.value.id == "self"
    }


def observed_attribute(engine: Any, name: str) -> Any:
    if name == REFUSED_INPUT:
        return getattr(getattr(engine, HOLDER_NAME, None), "refused_input", None)
    value = getattr(engine, name, {"absent": True})
    # The frontier fields are observed individually.
    if name == HOLDER_NAME and holder_type is not None and isinstance(value, holder_type):
        return {"absent": True}
    return value
