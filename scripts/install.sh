#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd -P)"

HERMES_HOME_DIR="${HERMES_HOME:-$HOME/.hermes}"
INSTALL_PROFILE="${HERMES_PROFILE:-default}"
export HERMES_HOME="$HERMES_HOME_DIR"
printf -v HERMES_COMMAND 'HERMES_HOME=%q hermes -p %q' "$HERMES_HOME_DIR" "$INSTALL_PROFILE"
if [[ "$INSTALL_PROFILE" != default ]]; then
  TARGET_ROOT="$HERMES_HOME_DIR/profiles/$INSTALL_PROFILE"
else
  TARGET_ROOT="$HERMES_HOME_DIR"
fi

PLUGIN_TARGET="$TARGET_ROOT/plugins/hermes-lcm-x"
SKILL_SOURCE="$REPO_ROOT/skills/hermes-lcm"
SKILL_TARGET="$TARGET_ROOT/skills/hermes-lcm-x"
CONFIG_FILE="$TARGET_ROOT/config.yaml"

# LCM-X 0.23.x and earlier installed as plugins/hermes-lcm (#471). Hermes
# matches plugins.enabled against the manifest name, not the directory, so an
# existing link to this checkout is reused instead of adding a second copy that
# would register the engine twice. The link setup never deletes files; host verbs own enablement.
LEGACY_PLUGIN_TARGET="$TARGET_ROOT/plugins/hermes-lcm"
LEGACY_SKILL_TARGET="$TARGET_ROOT/skills/hermes-lcm"
LEGACY_LEFTOVERS=()
LEGACY_PLUGIN_SEPARATE=0
if [[ -d "$LEGACY_PLUGIN_TARGET" ]]; then
  if [[ "$(cd "$LEGACY_PLUGIN_TARGET" && pwd -P)" == "$REPO_ROOT" ]]; then
    PLUGIN_TARGET="$LEGACY_PLUGIN_TARGET"
  else
    LEGACY_PLUGIN_SEPARATE=1
    LEGACY_LEFTOVERS+=("$LEGACY_PLUGIN_TARGET")
  fi
fi
if [[ -d "$LEGACY_SKILL_TARGET" ]]; then
  if [[ "$(cd "$LEGACY_SKILL_TARGET" && pwd -P)" == "$(cd "$SKILL_SOURCE" && pwd -P)" ]]; then
    SKILL_TARGET="$LEGACY_SKILL_TARGET"
  else
    LEGACY_LEFTOVERS+=("$LEGACY_SKILL_TARGET")
  fi
fi
LEGACY_CONFIG=0
if [[ -f "$CONFIG_FILE" ]] && grep -Eq \
  -e '(^|[^[:alnum:]_-])hermes-lcm([^[:alnum:]_-]|$)' \
  -e '^[[:space:]]*engine:[[:space:]]*["'"'"']?lcm["'"'"']?[[:space:]]*(#.*)?$' \
  "$CONFIG_FILE"; then
  LEGACY_CONFIG=1
fi

preflight_target() {
  local label="$1"
  local target="$2"
  local expected="$3"

  if [[ -L "$target" ]]; then
    local current_target
    # Compare canonical paths so a relative link to this checkout is reused.
    if ! current_target="$(cd -P "$target" 2>/dev/null && pwd -P)"; then
      current_target="$(readlink "$target")"
    fi
    if [[ "$current_target" != "$expected" ]]; then
      if [[ "$label" == "plugin" ]]; then
        echo "Refusing to replace existing symlink: $target -> $current_target" >&2
      else
        echo "Refusing to replace existing skill symlink: $target -> $current_target" >&2
      fi
      echo "Remove it manually or point it at this checkout before rerunning install.sh." >&2
      exit 1
    fi
  elif [[ -e "$target" ]]; then
    if [[ "$label" == "plugin" && -d "$target" ]]; then
      local physical_target
      physical_target="$(cd "$target" && pwd -P)"
      if [[ "$physical_target" == "$expected" ]]; then
        return
      fi
    fi
    if [[ "$label" == "plugin" ]]; then
      echo "Refusing to replace existing path: $target" >&2
    else
      echo "Refusing to replace existing skill path: $target" >&2
    fi
    echo "Move it aside or remove it manually before rerunning install.sh." >&2
    exit 1
  fi
}

preflight_target "plugin" "$PLUGIN_TARGET" "$REPO_ROOT"
preflight_target "skill" "$SKILL_TARGET" "$SKILL_SOURCE"

mkdir -p "$(dirname "$PLUGIN_TARGET")" "$(dirname "$SKILL_TARGET")"

if [[ ! -e "$PLUGIN_TARGET" && ! -L "$PLUGIN_TARGET" ]]; then
  ln -s "$REPO_ROOT" "$PLUGIN_TARGET"
fi
if [[ ! -e "$SKILL_TARGET" && ! -L "$SKILL_TARGET" ]]; then
  ln -s "$SKILL_SOURCE" "$SKILL_TARGET"
fi

# Core stays linked and dependency-free. Only the native install verb owns the
# companion's dependency-consent prompt; neither a TTY nor this script is consent.
COMPANION="hermes-lcm-x-local-embeddings"
COMPANION_SOURCE="electricsheephq/lcm-x/companions/$COMPANION"
if command -v hermes >/dev/null 2>&1; then
  if [[ "$LEGACY_PLUGIN_SEPARATE" == 0 ]]; then
    hermes -p "$INSTALL_PROFILE" plugins enable hermes-lcm-x
  else
    echo "Separate older copy preserved; migrate below before enabling hermes-lcm-x."
  fi
  # Both supported host pins accept repository subdirectories, not local dirs.
  # An explicit pinned install moves with install --force --ref, never update.
  COMPANION_TARGET="$TARGET_ROOT/plugins/$COMPANION"
  if ! LCM_X_REF="$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null)" || [[ ! "$LCM_X_REF" =~ ^[0-9a-f]{40}$ ]]; then
    echo "Cannot pin the optional companion without a Git commit; supply the core's published full commit SHA. Retry: $HERMES_COMMAND plugins install $COMPANION_SOURCE --ref <LCM-X-40-character-SHA>"
  elif [[ -L "$COMPANION_TARGET" ]]; then
    echo "Preserving companion symlink: $COMPANION_TARGET"
    echo "This host cannot replace an external symlink. Move only this link aside by hand,"
    echo "keeping its target intact, then run:"
    echo "  $HERMES_COMMAND plugins install $COMPANION_SOURCE --ref $LCM_X_REF"
  elif [[ -e "$COMPANION_TARGET" ]]; then
    echo "Companion already present; preserving $COMPANION_TARGET. To replace explicitly:"
    echo "  $HERMES_COMMAND plugins install $COMPANION_SOURCE --force --ref $LCM_X_REF"
  else
    if ! hermes -p "$INSTALL_PROFILE" plugins install "$COMPANION_SOURCE" --ref "$LCM_X_REF"; then
      echo "Optional companion install failed. Retry: $HERMES_COMMAND plugins install $COMPANION_SOURCE --ref $LCM_X_REF"
    fi
  fi
else
  echo "Hermes CLI unavailable; after installing Hermes and checking legacy migration, run: $HERMES_COMMAND plugins enable hermes-lcm-x"
  echo "Then run: $HERMES_COMMAND plugins install $COMPANION_SOURCE --ref <LCM-X-40-character-SHA>"
fi
echo "If the companion is disabled, run: $HERMES_COMMAND plugins enable $COMPANION"

cat <<EOF
Installed hermes-lcm-x at:
  $PLUGIN_TARGET

Discoverable skill:
  $SKILL_TARGET

Activation requires both:

plugins:
  enabled:
    - hermes-lcm-x

context:
  engine: lcm-x

Verification:
  1. Restart Hermes.
  2. Run: $HERMES_COMMAND plugins list
  3. Confirm the plugin list includes hermes-lcm-x and the selected context engine is lcm-x.
  4. Confirm the available skills include hermes-lcm.
EOF

if [[ "$LEGACY_CONFIG" == 1 || "$PLUGIN_TARGET" == "$LEGACY_PLUGIN_TARGET" || ${#LEGACY_LEFTOVERS[@]} -gt 0 ]]; then
  cat <<EOF

MIGRATION from hermes-lcm (LCM-X 0.23.x and earlier) - BREAKING in 0.24.0.
install.sh leaves a separate older copy's enablement unchanged and deletes nothing.
Run exactly one LCM copy, and change the config while Hermes is stopped:
  1. Stop Hermes.
  2. In $CONFIG_FILE replace hermes-lcm with hermes-lcm-x
     in plugins.enabled and set context.engine: lcm-x. The legacy
     context.engine: lcm still works but logs a deprecation warning.
EOF
  if [[ "$LEGACY_PLUGIN_SEPARATE" == 1 ]]; then
    cat <<EOF
     Do NOT keep both names enabled: a separate older copy is installed at
       $LEGACY_PLUGIN_TARGET
     and would load too (LCM-X then stays inert and is not running).
EOF
  elif [[ "$PLUGIN_TARGET" == "$LEGACY_PLUGIN_TARGET" ]]; then
    echo "     plugins/hermes-lcm is this checkout, so also keeping hermes-lcm listed is harmless."
  fi
  cat <<EOF
  3. Start Hermes; confirm '$HERMES_COMMAND plugins list' shows hermes-lcm-x enabled and the log shows
     "LCM plugin loaded — lossless context management active".
EOF
  if [[ ${#LEGACY_LEFTOVERS[@]} -gt 0 ]]; then
    echo "  4. Later, after verifying, remove the old copy by hand (keep it until"
    echo "     then for rollback):"
    printf '       %s\n' "${LEGACY_LEFTOVERS[@]}"
  fi
  cat <<EOF
If the config is not updated, Hermes logs "Context engine 'lcm' not found —
falling back to built-in compressor" and runs without LCM-X. The existing lcm.db
is untouched, but turns handled while Hermes runs without LCM-X are not in
lcm.db and their compacted content may not be recoverable. Update the config
before restarting Hermes after the update.
EOF
fi
