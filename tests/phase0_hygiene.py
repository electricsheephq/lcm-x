"""Derive product environment names without clearing test/benchmark controls."""

import ast
from functools import lru_cache
from pathlib import Path
import re


@lru_cache(maxsize=None)
def product_env_keys(root: Path) -> frozenset[str]:
    from hermes_lcm.config import ENV_FIELD_SPECS

    keys = {spec.env_key for spec in ENV_FIELD_SPECS}
    # Exact string literals include direct reads, parser arguments, and named
    # constants passed to getenv/environ.get. Root modules are product code;
    # deliberately do not recurse into tests/, bench/, or benchmarking/.
    for source in root.glob("*.py"):
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=source.name)
        keys.update(
            node.value for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
            and re.fullmatch(r"LCM_[A-Z0-9_]+", node.value)
        )
    return frozenset(keys)
