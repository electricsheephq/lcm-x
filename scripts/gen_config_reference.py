"""Regenerate the scalar environment reference: python scripts/gen_config_reference.py."""

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
# Direct execution starts with scripts/ on sys.path, rather than the checkout.
sys.path.insert(0, str(ROOT))

from config import ENV_FIELD_DEPRECATIONS, ENV_FIELD_SPECS, LCMConfig, _ENV_CHOICES, _ENV_RANGES

REFERENCE = ROOT / "skills/hermes-lcm/references/configuration.md"
START = "<!-- BEGIN GENERATED CONFIG REFERENCE -->"
END = "<!-- END GENERATED CONFIG REFERENCE -->"


def _code(value: object) -> str:
    text = json.dumps(value, ensure_ascii=False).replace("|", "&#124;").replace("`", "&#96;")
    return f"`{text}`"


def render_reference() -> str:
    """Render declared defaults only; never read environment or host configuration."""
    defaults = LCMConfig()
    lines = [
        "| Environment key | Type | Default | Diagnostic range / choices | Deprecation |",
        "| --- | --- | --- | --- | --- |",
    ]
    for spec in ENV_FIELD_SPECS:
        constraints = []
        if spec.name in _ENV_RANGES:
            minimum, maximum, inclusive = _ENV_RANGES[spec.name]
            constraints.append(f"`{'>=' if inclusive else '>'} {minimum}`")
            if maximum is not None:
                constraints.append(f"`<= {maximum}`")
        if spec.name in _ENV_CHOICES:
            constraints.extend(_code(choice) for choice in _ENV_CHOICES[spec.name][0])
        deprecation = "—"
        if spec.name in ENV_FIELD_DEPRECATIONS:
            since, removal = ENV_FIELD_DEPRECATIONS[spec.name]
            deprecation = f"Deprecated {since}; removed {removal}"
        lines.append(
            f"| `{spec.env_key}` | `{spec.py_type.__name__}` | "
            f"{_code(getattr(defaults, spec.name))} | {', '.join(constraints) or '—'} | {deprecation} |"
        )
    return "\n".join(lines) + "\n"


def update_reference(document: str) -> str:
    """Replace only the single marked section; malformed markers are an error."""
    if document.count(START) != 1 or document.count(END) != 1:
        raise ValueError("expected one configuration-reference marker pair")
    before, section = document.split(START)
    _, after = section.split(END)
    return before + START + "\n" + render_reference() + END + after


def main() -> None:
    document = REFERENCE.read_text(encoding="utf-8")
    updated = update_reference(document)
    if updated != document:
        REFERENCE.write_text(updated, encoding="utf-8")


if __name__ == "__main__":
    main()
