"""Sensitive-pattern policy and catalog for storage-boundary ingest protection."""

from __future__ import annotations

import hashlib
import re
from typing import Any, Sequence
from .ingest_content import (
    _safe_placeholder_metadata,
)
from .ingest_pem_model import (
    _normalize_escaped_solidus,
)
from .ingest_pem_scan import (
    _has_orphan_full_width_base64_run,
    _has_orphaned_private_key_body,
    _pem_fragment_near_private_key_placeholder,
    _pem_marker_with_inline_body,
    _redact_private_key_blocks_with,
)

_SENSITIVE_PLACEHOLDER_PREFIX = "[LCM sensitive redaction:"
_EMBEDDING_PRIVACY_PLACEHOLDER_PREFIX = "[LCM embedding privacy:"
# v3: single-pass PEM scanner (short-terminal-line bound, body-adjacent END
# pairing, linear on pathological inputs) — output differs from v2 on
# truncated/multi-key material, so v2-era vectors must re-embed.
_EMBEDDING_PRIVACY_TRANSFORM_VERSION = "privacy:v3"
_EMBEDDING_PRIVACY_PLACEHOLDER_RE = re.compile(
    r"\[LCM (?:sensitive redaction|embedding privacy):\s*"
    r"name=(?P<name>[a-z0-9_-]+)[^\]]*\]",
    re.IGNORECASE,
)
# Includes the resolver's aliases (embedding_provider.resolve_provider maps
# "openai"/"siliconflow" to the openai-compatible cloud provider): the
# privacy predicate must hold for every spelling a CALLER might pass, not
# just the canonical id the resolved provider object carries — a config
# string alias must never bypass cloud privacy (#370 review).
_EMBEDDING_PRIVACY_CLOUD_PROVIDERS = frozenset(
    {"voyage", "voyageai", "openai-compatible", "openai", "siliconflow"}
)
_SENSITIVE_PATTERN_CATALOG: dict[str, re.Pattern[str]] = {
    "api_key": re.compile(
        r"(?P<prefix>(?:\\?[\"']?)\b(?:api[_-]?key|api[_-]?token|access[_-]?token|secret[_-]?key|client[_-]?secret)\b\s*(?:\\?[\"']?)\s*[:=]\s*(?:\\?[\"']?))"
        r"(?P<secret>[A-Za-z0-9._~+/=-]{12,})"
        r"(?P<suffix>\\?[\"']?)",
        re.IGNORECASE,
    ),
    "bearer_token": re.compile(
        r"(?P<prefix>\bBearer\s+)"
        r"(?P<secret>[A-Za-z0-9._~+/=-]{12,})",
        re.IGNORECASE,
    ),
    "password_assignment": re.compile(
        r"(?P<prefix>\b(?:password|passwd|pwd|passphrase)\b\s*[\"']?\s*[:=]\s*)"
        r"(?:(?P<quote>[\"'])(?P<secret_quoted>[^\r\n\]\}]{6,}?)(?P=quote)|"
        r"(?P<secret_unquoted>[^\s,\"'\]}]{6,}))",
        re.IGNORECASE,
    ),
    "private_key": re.compile(
        r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----.*?-----END [A-Z0-9 ]*PRIVATE KEY-----",
        re.IGNORECASE | re.DOTALL,
    ),
}


def _redact_private_key_blocks(text: str) -> str:
    """Redact PEM private-key blocks with a linear scanner.

    This keeps large valid keys protected even when the optional ``regex``
    package is unavailable, without running the stdlib DOTALL private-key
    pattern over a pathological multi-MB input. A truncated block (BEGIN with
    no matching END) is redacted only through its contiguous PEM-shaped base64
    body, preserving unrelated trailing prose (#365).
    """
    return _redact_private_key_blocks_with(
        text, lambda secret: _sensitive_placeholder("private_key", secret)
    )


def sensitive_pattern_status(config: Any) -> dict[str, Any]:
    """Return metadata-only status for durable and provider-copy protection."""
    configured, active, unknown = _configured_sensitive_pattern_names(config)
    enabled = bool(getattr(config, "sensitive_patterns_enabled", False))
    privacy_setting = getattr(config, "embedding_privacy_enabled", None)
    privacy_provider = str(getattr(config, "embedding_provider", "") or "")
    privacy_required = embedding_provider_requires_privacy(privacy_provider)
    privacy_enabled = privacy_required and privacy_setting is not False
    return {
        "sensitive_patterns_enabled": enabled,
        "enabled": enabled,
        "sensitive_patterns": configured,
        "patterns": configured,
        "active_patterns": active if enabled else [],
        "unknown_patterns": unknown,
        "source": getattr(config, "sensitive_patterns_source", "default"),
        "embedding_privacy_setting": (
            "auto" if privacy_setting is None else "on" if privacy_setting else "off"
        ),
        "embedding_privacy_enabled": privacy_enabled,
        "embedding_privacy_provider_requires_privacy": privacy_required,
        "embedding_privacy_revision": (
            None
            if not privacy_required
            else "active" if privacy_enabled else "privacy:off"
        ),
        "placeholder_format": "[LCM sensitive redaction: name=<pattern>; chars=<n>; bytes=<n>; sha256=<16 for non-password>]",
        "lossless_recovery": False if enabled and active else None,
    }


def _configured_sensitive_pattern_names(config: Any) -> tuple[list[str], list[str], list[str]]:
    raw = getattr(config, "sensitive_patterns", []) or []
    if isinstance(raw, str):
        names = [part.strip() for part in raw.split(",") if part.strip()]
    else:
        names = [str(part).strip() for part in raw if str(part).strip()]
    if not names:
        return [], [], []
    configured: list[str] = []
    active: list[str] = []
    unknown: list[str] = []
    for name in names:
        normalized = name.lower().strip()
        if normalized in {"all", "default"}:
            for catalog_name in _SENSITIVE_PATTERN_CATALOG:
                if catalog_name not in configured:
                    configured.append(catalog_name)
                if catalog_name not in active:
                    active.append(catalog_name)
            continue
        configured.append(normalized)
        if normalized in _SENSITIVE_PATTERN_CATALOG:
            if normalized not in active:
                active.append(normalized)
        elif normalized not in unknown:
            unknown.append(normalized)
    return configured, active, unknown


class EmbeddingPrivacyPolicyError(RuntimeError):
    """Cloud embedding input cannot be proven safe under the active policy."""


class EmbeddingIdentityStaleError(EmbeddingPrivacyPolicyError):
    """The configured cloud privacy revision has no matching active vector identity (re-register with warmup)."""


def embedding_provider_requires_privacy(provider_id: str) -> bool:
    """Return whether a provider may send embedding input off-machine."""
    return str(provider_id or "").strip().lower() in _EMBEDDING_PRIVACY_CLOUD_PROVIDERS


def embedding_privacy_revision(config: Any) -> str:
    """Return the canonical identity revision for cloud embedding input.

    The active revision contains only a transform version and a digest of sorted
    pattern *names*. Explicit opt-out returns ``privacy:off``. Neither posture
    depends on matched text or secret bytes.
    """
    if getattr(config, "embedding_privacy_enabled", None) is False:
        return "privacy:off"
    configured, active, unknown = _configured_sensitive_pattern_names(config)
    if not configured or not active:
        raise EmbeddingPrivacyPolicyError(
            "cloud embedding privacy requires a nonempty sensitive-pattern policy"
        )
    if unknown:
        raise EmbeddingPrivacyPolicyError(
            "cloud embedding privacy policy contains unknown pattern names: "
            + ", ".join(sorted(unknown))
        )
    names = sorted(set(active))
    digest = hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest()
    return f"{_EMBEDDING_PRIVACY_TRANSFORM_VERSION}:{digest}"


def _embedding_privacy_placeholder(pattern_name: str) -> str:
    return (
        f"{_EMBEDDING_PRIVACY_PLACEHOLDER_PREFIX} "
        f"name={_safe_placeholder_metadata(pattern_name)}]"
    )


def _canonicalize_embedding_privacy_placeholders(text: str) -> str:
    return _EMBEDDING_PRIVACY_PLACEHOLDER_RE.sub(
        lambda match: _embedding_privacy_placeholder(
            str(match.group("name") or "unknown").strip().lower()
        ),
        text,
    )


def _embedding_privacy_redact_match(
    pattern_name: str, match: re.Match[str]
) -> str:
    group_names = match.re.groupindex
    secret_group = None
    for candidate in ("secret", "secret_quoted", "secret_unquoted"):
        if candidate in group_names and match.groupdict().get(candidate) is not None:
            secret_group = candidate
            break
    placeholder = _embedding_privacy_placeholder(pattern_name)
    if secret_group is None:
        return placeholder
    relative_start = match.start(secret_group) - match.start(0)
    relative_end = match.end(secret_group) - match.start(0)
    full = match.group(0)
    return full[:relative_start] + placeholder + full[relative_end:]


def _private_key_first(names: Sequence[str]) -> list[str]:
    """Order pattern names so private_key runs before any assignment pattern.

    An assignment pattern whose secret group matches the literal ``-----BEGIN``
    would otherwise consume the PEM begin marker and blind the private-key
    redactor, leaking the key body (#365). Relative order of the rest is kept.
    """
    ordered = [n for n in names if n == "private_key"]
    ordered.extend(n for n in names if n != "private_key")
    return ordered


def _embedding_privacy_redact_private_keys(text: str) -> str:
    """Replace complete or truncated PEM private-key blocks with placeholders."""
    return _redact_private_key_blocks_with(
        text, lambda _secret: _embedding_privacy_placeholder("private_key")
    )


def _embedding_privacy_residual_patterns(
    text: str, active_names: Sequence[str]
) -> list[str]:
    residual: list[str] = []
    for name in active_names:
        if name == "private_key":
            # Transform-INDEPENDENT check (#365): a surviving PEM END marker means
            # an earlier pattern consumed the BEGIN marker the redactor keys on, so
            # the key body shipped. Flag it even though the BEGIN-based redactor is
            # now blind. (BEGIN redaction still flagged for truncated/no-END blocks.)
            # Backstops (#383 rounds 4-6, #391): a surviving marker beside a
            # base64 run, >=2 contiguous full-width markerless base64 body
            # lines, and a 16+ base64 run in a private_key placeholder's
            # raw-text adjacency window — all block fail-closed, so the
            # validator no longer shares the transform's marker/placeholder-
            # keyed blind spot. The proximity backstop excludes pure-hex runs
            # (git SHAs / digests) so "redacted key + nearby git SHA in prose"
            # embeds (#389) — three line-model rewrites of it each re-opened a
            # leak shape, so the raw-text scan is authoritative (#391).
            # The marker-independent backstops scan for base64 runs; escaped
            # solidi (\/, any depth) split a run into sub-16 chunks and hide it,
            # so also run them on the solidi-normalized text (#383 round-7,
            # chwFQ). Placeholders and markers carry no solidi, so normalization
            # only rejoins base64 — strictly more sensitive, detection-only.
            normalized = _normalize_escaped_solidus(text)
            if (
                _embedding_privacy_redact_private_keys(text) != text
                or _has_orphaned_private_key_body(text)
                or _pem_marker_with_inline_body(text)
                or _pem_marker_with_inline_body(normalized)
                or _has_orphan_full_width_base64_run(text)
                or _has_orphan_full_width_base64_run(normalized)
                or _pem_fragment_near_private_key_placeholder(text)
                or _pem_fragment_near_private_key_placeholder(normalized)
            ):
                residual.append(name)
            continue
        if _SENSITIVE_PATTERN_CATALOG[name].search(text) is not None:
            residual.append(name)
    return residual


def _active_sensitive_pattern_names(config: Any) -> list[str]:
    if not bool(getattr(config, "sensitive_patterns_enabled", False)):
        return []
    _configured, active, _unknown = _configured_sensitive_pattern_names(config)
    return active


def _sensitive_placeholder(pattern_name: str, secret: str) -> str:
    parts = [
        f"{_SENSITIVE_PLACEHOLDER_PREFIX} "
        f"name={_safe_placeholder_metadata(pattern_name)}; "
        f"chars={len(secret)}; bytes={len(secret.encode('utf-8', errors='surrogatepass'))}"
    ]
    if pattern_name != "password_assignment":
        digest = hashlib.sha256(secret.encode("utf-8", errors="surrogatepass")).hexdigest()[:16]
        parts.append(f"sha256={digest}")
    return "; ".join(parts) + "]"


def _redact_match(pattern_name: str, match: re.Match[str]) -> str:
    group_names = match.re.groupindex
    secret_group = None
    for candidate in ("secret", "secret_quoted", "secret_unquoted"):
        if candidate in group_names and match.groupdict().get(candidate) is not None:
            secret_group = candidate
            break
    if secret_group is None:
        return _sensitive_placeholder(pattern_name, match.group(0))
    secret = match.group(secret_group)
    relative_start = match.start(secret_group) - match.start(0)
    relative_end = match.end(secret_group) - match.start(0)
    full = match.group(0)
    return full[:relative_start] + _sensitive_placeholder(pattern_name, secret) + full[relative_end:]


def _sensitive_pattern_for_key(key: Any, active_names: list[str]) -> str | None:
    if not isinstance(key, str):
        return None
    normalized = re.sub(r"[^a-z0-9]+", "_", key.lower()).strip("_")
    compact = normalized.replace("_", "")
    if "api_key" in active_names and (
        compact in {"apikey", "apitoken", "accesstoken", "secretkey", "clientsecret"}
        or ("api" in normalized and "key" in normalized)
        or ("access" in normalized and "token" in normalized)
        or ("secret" in normalized and "key" in normalized)
    ):
        return "api_key"
    if "bearer_token" in active_names and compact in {"authorization", "authtoken", "bearertoken", "token"}:
        return "bearer_token"
    if "password_assignment" in active_names and compact in {"password", "passwd", "pwd", "passphrase"}:
        return "password_assignment"
    return None


def _redact_entire_sensitive_string(text: str, pattern_name: str) -> str:
    if not text or _SENSITIVE_PLACEHOLDER_PREFIX in text:
        return text
    return _sensitive_placeholder(pattern_name, text)
