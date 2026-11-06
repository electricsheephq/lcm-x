"""Stable config observations: pin old defaults, omit new fields at default."""
import dataclasses
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from hermes_lcm.config import LCMConfig


BASELINE = json.loads((Path(__file__).parents[1] / "fixtures" / "trace_config_baseline.json").read_text())


def trace_config(config: LCMConfig, serialize: Callable[[Any, str], Any]) -> dict[str, Any]:
    """Use each harness's value serializer without changing other dataclasses.

    The committed baseline is deliberately never refreshed from live defaults:
    changing a known default must remain visible. A field missing from the
    baseline compares against a fresh LCMConfig(), so a new field at its default
    moves no golden; test_baseline_lists_every_config_field then requires the
    same PR to pin that default.
    """
    result = {"__trace_config__": BASELINE["version"]}
    current_defaults = None
    for field in dataclasses.fields(config):
        name = field.name
        value = serialize(getattr(config, name), name)
        if name in BASELINE["defaults"]:
            default = BASELINE["defaults"][name]
        else:
            if current_defaults is None:
                current_defaults = LCMConfig()
            default = serialize(getattr(current_defaults, name), name)
        if value != default:
            result[name] = value
    return result
