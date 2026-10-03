# Packaging and distribution posture

## Current decision

LCM-X intentionally remains a clone-or-symlink Hermes user plugin for now. From
v0.24.0 the plugin name and install directory are `hermes-lcm-x` and the context
engine is `lcm-x` (#471; breaking for configs that enable only `hermes-lcm` — see
the operator guide's migration section). The supported install path is:

```bash
git clone https://github.com/electricsheephq/lcm-x \
  ~/.hermes/plugins/hermes-lcm-x
```

For profile-specific installs, clone under `~/.hermes/profiles/<profile>/plugins/hermes-lcm-x`. For development checkouts, `scripts/install.sh` creates a profile-aware symlink into the active Hermes plugin directory and refuses to overwrite an existing checkout or unrelated symlink.

## Why not pip-style packaging yet?

The repository is a Hermes plugin, not a standalone Python application. Runtime discovery currently depends on:

- `plugin.yaml` declaring the plugin name, registered tools, and the required runtime dependency `numpy>=2.2,<3`
- the repo root containing `__init__.py` for Hermes plugin registration
- the operator placing or symlinking the checkout into Hermes' plugin search path
- Python 3.11+; semantic/hybrid retrieval extras remain optional (see [`requirements-semantic.txt`](../requirements-semantic.txt))

Clone/symlink distribution and Hermes package-manager membership are separate concerns. The checkout remains the plugin distribution unit, while Hermes' supported package manager reads the manifest-only NumPy declaration and stages it as a generated dependency-only workspace member. The checkout itself remains free of Python package metadata.

The repository deliberately has no committed `pyproject.toml`. Package metadata waits until Hermes plugin packaging/discovery has a stable target for pip-installed plugins: adding generic Python packaging before the host install contract is clear would create a second install story without making first-run activation simpler.

Lint settings live in `ruff.toml`, not a lint-only `pyproject.toml`. Hermes' package manager (`pm/`, after v2026.9.24) treats any checkout with a `pyproject.toml` as a uv workspace member. It adds a `[project]` name but a lint-only file has no `version`, so `uv lock` rejects it and the update can add `hermes-lcm-x` to `plugins.disabled` (#631). Do not add a `pyproject.toml` back just for tooling. `tests/test_issue_631_not_a_pm_member.py` guards the lint-only regression; `tests/test_packaging_install.py` verifies the supported manifest-dependency staging path with the real Hermes manager.

## Package-manager CI contract

The existing `test (3.14)` CI lane always runs the real-manager staging test
against Hermes commit `0a374d167424cdc730ce9761368b62255b551e58`.
That pinned PM declares Python 3.14 as its runtime; the plugin's other supported
Python versions continue through the ordinary unit-test matrix. CI installs the
PM's own pinned dependencies from `pm/pyproject.toml` and keeps its reference tree
outside the plugin checkout so it cannot contaminate test discovery or compilation.

The CI step sets `LCM_REQUIRE_REAL_HERMES=1`, `LCM_REAL_HERMES_SRC`, and
`LCM_REAL_HERMES_PYTHON`. Missing prerequisites fail that step rather than skip it.
Local runs may still opt in with the latter two variables; without them and without
the required flag, the integration test explicitly skips. The test checks actual
declaration parsing and dependency-only workspace staging, not a complete install
or application restart.

## Next packaging step

Make packaging a separate implementation lane only when one of these is true:

1. Hermes Agent documents a stable pip/distribution entrypoint for plugins.
2. Users need version-pinned installs without direct git checkouts.
3. Release automation needs packaged artifacts beyond GitHub tags/releases.

The narrow next step would be packaging metadata plus tests that prove a packaged install still exposes `hermes-lcm-x`, context engine `lcm-x`, and all 15 LCM tools through `hermes plugins`. Until then, clone/symlink remains the documented path.

## Current install and update references

- Quickstart: [README](../README.md)
- Detailed install/update/verify: [Operator guide](operator-guide.md)
- Standalone install script contract: [`tests/test_packaging_install.py`](../tests/test_packaging_install.py)
