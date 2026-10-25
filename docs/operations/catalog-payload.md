# Catalog payload

Catalog installs use a generated runtime payload under `plugin/` on the append-only
`dist` branch. Source checkouts and `scripts/install.sh` retain their existing layout.
The payload is the conservative AST import closure of root `__init__.py` plus the
reviewed assets in `release/payload-assets.txt`. Conditional and lazy imports count.
No development trees ship. Adding a runtime asset requires reviewing the allowlist.

The **Publish catalog payload** workflow builds from a `v*` tag's resolved commit.
Manual dispatch accepts an existing source tag, including historical backfills. It
uses the workflow token with `contents: write`, creates the first dist commit without
a parent, and makes every subsequent commit a child of the current dist head.
Pushes are ordinary fast-forward pushes. A rejection stops publication; the release
manager must inspect the reported rule and change settings if appropriate. The
workflow never bypasses a ruleset or retries with force.
Publication runs queue without cancellation (GitHub's finite limit is 100 pending runs).

The builder renders `plugin/README.md` from the reviewed
`scripts/payload/README.md.tmpl`. It contains catalog install/enable commands and
absolute source-SHA-pinned links, and is hashed in the manifest like every asset.
Builder version 2 adds this README and the `tooling_dirty` manifest field.
In SHA mode, the builder omits template links absent from the source commit and
records their source paths in `readme_omitted_links` (empty when none are omitted).

Each dist commit contains `plugin/` and `payload-manifest.json`. The commit message
is `payload: <tag> <source-sha>`; the workflow prints the dist SHA and attaches the
same manifest to the existing GitHub release. The release must be published (not a draft) before any dist
build or push; the workflow polls every 10 seconds for at most 10 minutes. If it
times out, nothing is pushed; re-run with `workflow_dispatch` once the release is published.
Repeating a tag/source pair reuses its existing dist commit and re-attaches that commit's stored manifest,
even when dispatch tooling has changed. An upload failure can be retried this way.

## Verify provenance

Resolve the source tag to its commit and compare it to `source_sha` in the manifest.
Use the builder and allowlist at the manifest's `tooling_sha` (which may differ from
the source SHA for backfills), and verify `allowlist_sha256`. From that clean
checkout, build into a fresh output directory:

```sh
python scripts/build_plugin_payload.py --source <source-sha> --out <fresh-output> --manifest <rebuilt-manifest.json>
```

Compare the rebuilt manifest with both the release asset and the manifest at the
exact dist SHA. Compare every sorted path, SHA-256, byte count, total, file count,
builder version, tooling SHA and allowlist hash. Hash the dist `plugin/` files as well; a manifest alone does
not prove the dist tree matches. Builder limits mirror the host at
[`6f7a7991`, `plugin_guard.py:102-104`](https://github.com/NousResearch/hermes-agent/blob/6f7a7991bb069db07ae74a479823ce8310f8c7e0/tools/plugin_guard.py#L102):
400 files, 10 MiB total, and 1 MiB per file. Symlinks are rejected, including links
outside the source tree. Every allowlist entry must match a tracked file.
SHA mode refuses uncommitted changes in the tooling checkout, including untracked
files. For hand runs only, `--allow-dirty-tooling` permits these changes and records
`tooling_dirty: true` in the manifest; clean builds record `false`. Reproducible
provenance verification requires clean tooling. CI uses a fresh tooling checkout.
Checkout mode refuses staged or unstaged tracked changes. Untracked receipts are
ignored and never selected. The drift gate rescans payload Python for reviewed
file-relative and dynamic-import sites in `tests/fixtures/payload-reviewed-sites.json`;
new or removed sites require review of the committed baseline.
The payload link check reads text as UTF-8. Invalid encoding and NUL bytes in
text files fail with the file name, while binary assets are skipped.

## Catalog pin and rollback

The catalog entry uses the source repository URL, the immutable **dist commit SHA**
as `sha`, and `subdir: plugin`. Its version describes the source release. Do not
use the source commit as the payload pin or send source-installer users to `dist`.
Rollback re-pins the catalog to a retained previous dist commit, with its matching
version and `subdir`. Never rewrite or remove dist history.
Find the installed dist commit with `hermes plugins list` (short pin), or the full
`revision` in `$HERMES_HOME/plugins/.install-metadata.json`.
For a dependency companion, use the **source SHA**, not the installed dist revision:
read it from `payload: <tag> <source-sha>` on the exact dist commit, or `source_sha`
in that commit's manifest / matching release asset. The catalog installation itself
does not expose this source SHA.

## Scanner and consent boundary

The scanner remains enabled. The committed medium-or-higher baseline in
`tests/fixtures/catalog-scanner-findings.json` includes `proc_access` (high) in
`cpu_capacity.py` and `python_subprocess` (medium) in `inactive_record.py` and
`runtime_identity.py`, with occurrence counts (2, 1 and 1 respectively). Added or
removed medium-or-higher occurrences fail the real-host cell; critical findings fail.
The CPU detector's read-only proc discovery currently produces CAUTION; no path
literal is disguised to change that verdict. Historical 0.28.1 and rc1 payloads
scan SAFE with the measured upstream scanner.

The real-host cell uses isolated local Git transport and the ordinary catalog,
scanner, checkout, revision metadata and engine loader. It proves those mechanics
on the two pinned hosts; it does not prove network distribution or publication.
Adding tools can require explicit catalog repin consent. The 0.28.1-to-candidate
unattended repin must raise `RepinConsentRequired` and preserve the installed
revision. A successful reviewed catalog install does not prove a consent-free
update; CAUTION may also require confirmation on other update paths.
