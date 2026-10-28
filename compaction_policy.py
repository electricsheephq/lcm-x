"""Pure context-window, compaction-trigger and active-assembly policy.

The engine owns parsing, live environment reads, ordered state writes and logs.
Its shells compose these pure helpers in the required order.
"""

from __future__ import annotations

from collections.abc import Mapping

from .codex_routing import _codex_oauth_context_cap, _is_codex_gpt55_route

_CODEX_GPT55_COMPACTION_THRESHOLD = 0.85


def resolve_threshold(
    configured: float,
    source: str,
    model_thresholds: Mapping[str, float],
    autoraise_enabled: bool,
    model: str | None,
    provider: str | None,
) -> tuple[float, str, dict[str, float] | None]:
    """Resolve longest model override, then route auto-raise, then configuration."""
    configured = float(configured)
    explicit_lcm_override = source in {
        "env:LCM_CONTEXT_THRESHOLD",
        "config_yaml:lcm.context_threshold",
    }
    if model_thresholds and model:
        best_key = ""
        for key in model_thresholds:
            if key in model and len(key) > len(best_key):
                best_key = key
        if best_key:
            override = float(model_thresholds[best_key])
            return override, f"model_thresholds:{best_key}", {"from": configured, "to": override}
    # 5.6 deliberately excluded: extending auto-raise requires its own retention run.
    if (
        _is_codex_gpt55_route(model, provider)
        and autoraise_enabled
        and not explicit_lcm_override
        and configured < _CODEX_GPT55_COMPACTION_THRESHOLD
    ):
        return (
            _CODEX_GPT55_COMPACTION_THRESHOLD,
            "codex_gpt55_autoraise",
            {"from": configured, "to": _CODEX_GPT55_COMPACTION_THRESHOLD},
        )
    return configured, source, None


def effective_context_length(
    raw_context_length: int, model: str | None, provider: str | None,
) -> tuple[int, int | None, str]:
    cap = _codex_oauth_context_cap(model, provider)
    if cap is not None and raw_context_length > cap:
        return cap, cap, "codex_oauth_context_cap"
    return raw_context_length, None, ""


def assembly_token_cap(
    context_length: int, max_assembly_tokens: int, reserve_tokens_floor: int,
) -> tuple[int | None, bool]:
    """Return the cap and whether the reserve floor disables its own constraint."""
    caps: list[int] = []
    reserve_floor_disabled = False
    if max_assembly_tokens > 0:
        caps.append(max_assembly_tokens)
    if context_length > 0 and reserve_tokens_floor > 0:
        reserve_cap = context_length - reserve_tokens_floor
        if reserve_cap > 0:
            caps.append(reserve_cap)
        else:
            reserve_floor_disabled = True
    return (max(1, min(caps)) if caps else None), reserve_floor_disabled


def threshold_tokens(
    context_threshold_tokens: int,
    assembly_cap: int | None,
) -> int:
    """Expose the stricter assembly trigger."""
    if assembly_cap is not None and assembly_cap > 0:
        if context_threshold_tokens > 0:
            return min(context_threshold_tokens, assembly_cap)
        return assembly_cap
    return context_threshold_tokens
