"""The host side of the live-rows producer: a minimal `agent` package registered in sys.modules.

The repository ignores every `agent/` directory (CI writes its own host stub there), so the modules ship under this
name and `install()` registers them as `agent`, `agent.context_engine` and `agent.auxiliary_client` before a tree
is imported. Nothing here needs the Hermes source.
"""

from __future__ import annotations

import sys
import types

from . import auxiliary_client, context_engine


def install() -> types.ModuleType:
    package = types.ModuleType("agent")
    package.__path__ = []  # a package, so `from agent import auxiliary_client` resolves the attribute below
    package.context_engine = context_engine
    package.auxiliary_client = auxiliary_client
    sys.modules["agent"] = package
    sys.modules["agent.context_engine"] = context_engine
    sys.modules["agent.auxiliary_client"] = auxiliary_client
    return package
