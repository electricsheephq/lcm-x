"""Plugin and context-engine identity for LCM-X, plus the legacy-name migration notice.

LCM-X installs as plugin ``hermes-lcm-x`` with context engine ``lcm-x`` (#471).
Hermes matches ``context.engine`` against the registered engine's ``name``, so a
config that still says ``context.engine: lcm`` keeps selecting this engine
through the legacy alias below. The alias is loud: a once-per-process warning
plus an ``identity_migration`` field on ``lcm_status`` / ``lcm_doctor`` that
names the exact config to change.

``plugins.enabled`` cannot be aliased from here: Hermes matches it against the
manifest ``name`` before any plugin code runs. A config that enables only
``hermes-lcm`` never loads this plugin, and Hermes logs
``Context engine 'lcm' not found — falling back to built-in compressor``.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Mapping

logger = logging.getLogger(__name__)

PLUGIN_NAME = "hermes-lcm-x"
ENGINE_NAME = "lcm-x"
LEGACY_PLUGIN_NAME = "hermes-lcm"
LEGACY_ENGINE_NAME = "lcm"
# Every name this engine answers to when Hermes (or LCM itself) compares
# ``engine.name``.
ENGINE_NAMES = frozenset({ENGINE_NAME, LEGACY_ENGINE_NAME})

_warning_lock = threading.Lock()
_warning_emitted = False


def is_lcm_engine_name(name: Any) -> bool:
    """True when *name* is the current or legacy LCM-X engine name."""
    return isinstance(name, str) and name in ENGINE_NAMES


def identity_migration_notice(hermes_config: Any) -> dict[str, Any] | None:
    """Return the migration notice for a Hermes config, or None when it is current.

    Fires when ``context.engine`` is the legacy ``lcm`` or ``plugins.enabled``
    still lists the legacy ``hermes-lcm`` name.
    """
    if not isinstance(hermes_config, Mapping):
        return None
    context_cfg = hermes_config.get("context")
    configured_engine = (
        context_cfg.get("engine") if isinstance(context_cfg, Mapping) else None
    )
    plugins_cfg = hermes_config.get("plugins")
    enabled = plugins_cfg.get("enabled") if isinstance(plugins_cfg, Mapping) else None
    enabled_names = (
        [str(item) for item in enabled] if isinstance(enabled, (list, tuple)) else []
    )

    legacy_engine = configured_engine == LEGACY_ENGINE_NAME
    legacy_enabled = LEGACY_PLUGIN_NAME in enabled_names
    if not (legacy_engine or legacy_enabled):
        return None

    changes: list[str] = []
    if legacy_engine:
        changes.append(
            f"set `context.engine: {ENGINE_NAME}` (currently `{LEGACY_ENGINE_NAME}`, "
            "accepted as a deprecated alias)"
        )
    if PLUGIN_NAME not in enabled_names:
        changes.append(f"add `{PLUGIN_NAME}` to `plugins.enabled`")
    if legacy_enabled:
        changes.append(
            f"remove `{LEGACY_PLUGIN_NAME}` from `plugins.enabled` once "
            f"`{PLUGIN_NAME}` is enabled"
        )
    # The alias is fixed when register() runs (#477): a config edited while
    # Hermes runs is not picked up, so the steps lead with the stop/restart.
    steps = ["stop Hermes", *changes, "start Hermes again"]
    return {
        "status": "deprecated_config",
        "plugin_name": PLUGIN_NAME,
        "engine_name": ENGINE_NAME,
        "configured_engine": configured_engine,
        "legacy_engine_alias_active": legacy_engine,
        "legacy_plugin_enabled": legacy_enabled,
        "change": steps,
        "message": (
            "Stop Hermes, then edit config.yaml: " + "; ".join(changes) + "; "
            "then start Hermes again. Editing config.yaml while Hermes runs makes "
            "new sessions fall back to the built-in compressor until Hermes "
            f"restarts. LCM-X was renamed: plugin `{LEGACY_PLUGIN_NAME}` -> "
            f"`{PLUGIN_NAME}`, context engine `{LEGACY_ENGINE_NAME}` -> `{ENGINE_NAME}`. "
            f"A config that enables only `{LEGACY_PLUGIN_NAME}` stops loading LCM-X "
            "and Hermes falls back to its built-in compressor (lcm.db is untouched)."
        ),
    }


def warn_identity_migration_once(notice: dict[str, Any] | None) -> bool:
    """Log *notice* as a warning once per process. Returns True when it logged."""
    global _warning_emitted
    if not notice:
        return False
    with _warning_lock:
        if _warning_emitted:
            return False
        _warning_emitted = True
    logger.warning("DEPRECATED LCM-X config: %s", notice["message"])
    return True


def load_hermes_config() -> Any:
    """Best-effort read of the active Hermes profile config (None when unavailable)."""
    try:
        from hermes_cli.config import load_config
    except Exception:
        return None
    try:
        return load_config()
    except Exception as exc:
        logger.warning(
            "LCM-X could not read the Hermes config to check for legacy names: %s", exc
        )
        return None
