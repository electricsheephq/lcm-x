# S1-C.3 qualification delta
Product consumers unchanged from S1-C.2 below. Additional test-only readers: dependency_probe.py:31-66 reads the synthetic scanner policy, core/companion tree, exact PM lock entries and fetch archives; dependency_qualification.py:192-195 admits only the hosted locked-tool preflight and :207-235 scopes/restores synthetic scanner config. No new installer/runtime consumer.

# S1-C.2 current implementation inventory

Candidate: `dda050688c53d14db533e010bfcb044b244df206`, core manifest unchanged.
The companion path is `companions/hermes-lcm-x-local-embeddings/`; stable version
0.1.0 and one dependency range. No executable runtime reads requirements-semantic.txt.
The installer reads its checkout HEAD for --ref and passes the companion repository
subdirectory to the native host installer. It never links the companion.

## LCM-X readers

| Reader | Meaning |
| --- | --- |
| `tests/test_issue_1186_dependency_companion.py:16-27` | Core dependency exclusion, carrier identity/version/fields and requirements range consistency |
| `tests/test_issue_1080_catalog_asks.py:73` | Requirements range guard |
| `tests/test_issue_631_not_a_pm_member.py:42-44` | Existing dependency-free core/PM membership invariant retained |
| `scripts/install.sh:102-125` | Companion source/name/path, pin and native install/enable recovery hints |
| `embedding_provider.py:2105-2110` | Missing-provider health detail consumed by doctor/startup |
| `diagnostics.py:214-220` | Doctor operator install/enable advice |
| `tests/test_issue_1186_dependency_companion.py:30-91` | Installer command/consent boundary and remediation regression readers |
| `tests/fixtures/command_doctor.json:35`, `tests/fixtures/lcm_doctor.json:7709` | Exact doctor public text/JSON expectations |
| `docs/embeddings-setup.md:71-73,106,120,131` | Native install, later enable, explicit pinned replacement and last-resort requirements example |
| `requirements-semantic.txt:4,6,9` | Operator command references; only the last-resort example consumes this file |
| `CHANGELOG.md:294` and `.github/release-notes/v0.24.9{,-rc1,-rc2,-rc3,-rc4}.md` | Historical text readers only; unchanged |
| `companions/hermes-lcm-x-local-embeddings/README.md:1-11` | Companion lifecycle explanation; links setup guide |

Complete lexical inventory saved as `raw/r2-consumers-grep.txt`.

Qualification readers added in this candidate:
- `bench/instruments/reliability/dependency_qualification.py:25-26,185,212,270,284,303`: companion identity/source, exact pins, installed manifest, core version-only edit and dependency-range edit.
- `bench/instruments/reliability/dependency_probe.py:56-101`: actual PluginManager core/carrier selection, import, full-text recall and doctor detail.
- `tests/test_issue_1186_host_dependencies.py:16-56`: synthetic prompt/measurement checks and hosted integration gate.
- `.github/workflows/real-host.yml:91-169`: exact host/route arms, raw uploads and S1-C.3 hosted-only locked-tool-download flag (on).

## Exact-host functional readers

Both full public source checkouts were freshly fetched at the exact pins, not
inferred from the scope's seed venv. Raw field-reader inventories (including
unrelated profile-manifest matches, clearly only lexical hits) are
`raw/r2-pm-consumers.txt` and `raw/r2-reference-consumers.txt`.

| Host / reader | Meaning |
| --- | --- |
| PM `6f7a7991`, `pm/plugin_declarations.py:10,110-156` | Reads plugin YAML and effective dependency aliases; pyproject precedence |
| PM `pm/workspace.py:42-67,172,226,254` | Selects dependency members; hashes manifest bytes; emits virtual dependency-only members |
| PM `pm/packages.py:387-389` | Incorporates member stamp into shared-environment currency |
| PM `pm/plugins_state.py:93` | Shared live-home/profile union; no per-profile copy of dependency graph |
| PM `hermes_cli/plugins_cmd_install.py:43-46,84-128,283-525` | Native subdirectory install, effective dependency consent, disabled-on-decline and atomic enable |
| PM `hermes_cli/plugins_transaction.py:40-76`, `plugins_cmd_update.py:23-67` | Unattended dependency-change gate, recorded subdirectory replacement; stable core bypasses hard dependency gate |
| PM `hermes_cli/plugins_cmd.py:469-588,636-685`, `plugins_admission.py:50` | Explicit enable/selection admission (host treats explicit enable as consent) |
| PM `hermes_cli/plugins_manifest.py:150-181,456,487-503` | Name/version/description and python_dependencies parser/model |
| Both pins `hermes_cli/plugins_loader.py:372-390,444-496,630-645` | Warns on missing dependencies and requires __init__.py with register(ctx); explains minimal no-op entry point |
| Both pins `hermes_cli/plugin_dev.py:195-196,301-322`, `plugin_validate.py:538` | Development/validation dependency diagnostics |
| PM `hermes_cli/plugins_cmd_catalog.py:483-501`, `plugins_cmd_install.py:596`, `plugins_cmd_update.py:294` | Catalog/install/update JSON dependency summaries |
| PM `hermes_cli/plugins.py:1553-1554`, reference same file `1511-1512` | General list/registry projection of name/version/description; other registration readers in raw inventories are inert because carrier registers nothing |
| Both pins `hermes_cli/plugins_discovery.py:178-281`, `plugins_cmd.py` lookup helpers, `plugins_ledger.py:197,259` | Name/key matching, discovery/selection and ownership bookkeeping |
| PM `pm/plugin_eviction.py:51-69`, `pm/publication.py:53-56` | Manifest compatibility validation during generation/transaction publication |
| Reference `f97608f1`, `hermes_cli/plugin_python_deps.py:36,94,213,234,278,304,365,423` | Legacy effective requirements, enabled-home union, install/reapply and update persistence |
| Reference `hermes_cli/plugins_cmd.py:225-261,693-729,916-983,1329-1369` | Native subdirectory clone and install, explicit enable before legacy dependency preparation; no dependency consent question |
| Reference `hermes_cli/plugins_cmd.py:1124,2112,2235`, `plugins_cmd_catalog.py:410-412`, `plugin_packs.py:394` | Additional update/catalog/pack legacy dependency-install consumers |
| Reference `hermes_cli/update_cmd_deps.py:495,1238-1318`, `update_cmd.py:658,759`, `update_cmd_zip.py:413` | Normal host update reapplication of recorded plugin requirements |
| Both pins `hermes_cli/web_server_memory.py`, `web_routers/memory_providers.py` | Generic memory-provider setup dependency readers (carrier is not a memory provider and does not register that surface) |

Source identity links:
[PM host](https://github.com/NousResearch/hermes-agent/tree/6f7a7991bb069db07ae74a479823ce8310f8c7e0),
[reference host](https://github.com/NousResearch/hermes-agent/tree/f97608f178d1ffeca59860195ab7da295f7c8e5f).
This is consumer/source evidence, not an installed-host qualification pass.

---

# #1186 slice 1 dependency consumers

## S1-C resumption, 2026-10-24

Re-read at `df9e8b490747b48e2c88d64d0362ebfa576d561a` after the required
fast-forward. S1-C supersedes option A below: core remains dependency-free;
the intended companion is `companions/hermes-lcm-x-local-embeddings/`.
No companion or product edit was made because the linked-companion enable
route lacks the required host-owned dependency-consent question.

Updated exact-name inventory: [raw/s1c-consumers-grep.txt](raw/s1c-consumers-grep.txt).
The requirements readers below remain the same, except `CHANGELOG.md` now
references the file at line 294. There are no existing companion-path readers.
`tests/test_issue_631_not_a_pm_member.py:42-44` agrees with S1-C and must stay.

Additional verified upstream-main readers (same exact `6f7a7991` pin):

| Reader | Role |
| --- | --- |
| `hermes_cli/plugins_cmd.py:280-316` | Parses native subdirectory plugin identifiers, including owner/repo/subdir |
| `hermes_cli/plugins_cmd.py:636-685` | Linked-plugin enable; selection admission occurs before capability grants, with no Python-dependency consent call |
| `hermes_cli/plugins_cmd_install.py:283-409` | Native install resolves the companion subdirectory and publishes its manifest; replacing an existing target requires force |
| `hermes_cli/plugins_transaction.py:40-76` | Dependency-change gate for updates; returns immediately for disabled plugins |
| `hermes_cli/plugins_transaction.py:17-37` | Reinstall consent for active replacements, followed by PM publication |
| `hermes_cli/plugins_cmd_update.py:23-67` | Native subdirectory re-clone/update route; not a consented enable verb for an unrecorded symlink |
| `hermes_cli/plugins_cmd_catalog.py:767-792` | `validate --install-deps` submits PM candidates without a dependency-consent question; unsuitable as an installer workaround |

Reference `f97608f1`, `hermes_cli/plugins_cmd.py:225-261,693-729` also parses
and checks out subdirectory identifiers. This is source evidence only, not an
executed native-install pass. Its legacy no-prompt enable semantics are accepted
by S1-C and are not the current blocker.

Git blob verification for new upstream source: [raw/s1c-enable-consent-probe.json](raw/s1c-enable-consent-probe.json).
The original r0 inventory follows as historical evidence; its recommendation to
retarget tests to option A is superseded and must not be implemented.

## Original r0 inventory (superseded decision, preserved evidence)

Status: preflight inventory at LCM-X `efc32d53291fbd4955c5c7e63c40770a38b400ec`.
The host-change stop fired before implementation. **No manifest field was touched.**
This is a complete exact-name reference inventory for `requirements-semantic.txt`
in the LCM-X tree, plus the inspected consumers of the proposed manifest dependency
field. It is not a claim of completed cross-host qualification or an exhaustive
inventory of every generic manifest pass-through in the host repositories.

## requirements-semantic.txt

Search: `git grep -n -e requirements-semantic -e python_dependencies -e pip_dependencies`.
Raw result: [raw/consumers-grep.txt](raw/consumers-grep.txt).
No tracked executable installer/runtime reader of this requirements file was found.

| Consumer | Role |
| --- | --- |
| `requirements-semantic.txt:11` | Self-referential manual installation example |
| `docs/embeddings-setup.md:71` | Installation example |
| `docs/embeddings-setup.md:105` | Host-update recovery example |
| `tests/test_issue_1080_catalog_asks.py:73` | Reads the requirements file and checks its range |
| `CHANGELOG.md:250` | Historical declaration/recovery description |
| `.github/release-notes/v0.24.9-rc1.md:29` | Historical release description |
| `.github/release-notes/v0.24.9-rc2.md:28` | Historical release description |
| `.github/release-notes/v0.24.9-rc3.md:16` | Historical release description |
| `.github/release-notes/v0.24.9-rc4.md:14` | Historical release description |
| `.github/release-notes/v0.24.9.md:10` | Historical release description |

## Proposed python_dependencies field and legacy alias

LCM-X consumer: `tests/test_issue_631_not_a_pm_member.py:42-44` reads top-level
manifest keys and checks for `python_dependencies` / `pip_dependencies`. Its old
non-membership expectation would need reconciliation with the explicit option A
ruling when implementation resumes. No implementation adjustment was made.

Pinned source snapshots and untruncated Git trees are under `raw/hosts/` and
`raw/*-tree.json`; the source links below bind the actual inspected revisions.

| Host and reader | Role / downstream consumers inspected |
| --- | --- |
| upstream-main `6f7a7991`, `pm/plugin_declarations.py:110,149` | `read_python_declaration` combines the two manifest dependency keys when no real project declaration takes precedence; `PythonDeclaration.install_requirements:67`, `is_member:71` govern dependency membership |
| same pin, `hermes_cli/plugins_cmd_install.py:43,46,84,125,128` | Reads the declaration for install consent and dependency summary; `_consent_python_deps` refuses a noninteractive or declined install |
| same pin, `hermes_cli/plugins_cmd_install.py:478,486,505` | Installation reads consent, refuses activation on decline, and only then calls enable admission |
| same pin, `hermes_cli/plugins_cmd.py:469,541,588,636` | Shared selection/admission path; `cmd_enable` is the linked-plugin activation command, **not a dependency-consent prompt** |
| same pin, `hermes_cli/plugins_admission.py:50` | Delegates candidate selection to `pm.client.sync_venv`; config and environment are published together |
| reference `f97608f1`, `hermes_cli/plugin_python_deps.py:36,94,114` | Reads both manifest dependency aliases into the legacy declaration |
| same reference, `hermes_cli/plugin_python_deps.py:213,234,278,304,365,423` | Enabled declarations, profile-home union, candidate check, direct install and host-update reapply; not versioned PM generations |
| same reference, `hermes_cli/plugins_cmd.py:376,385,392,983,1329,1359,1369` | Quiet/install/enable dependency consumers; enable mutates selection and activates before the dependency-install helper; no dependency-consent prompt in that helper |

Line numbers identify function starts or specific call sites; see the snapshotted
source for surrounding control flow. Only source checked during preflight is
listed. Resume must complete the full host-consumer search if fields are changed.

Sources:
- [LCM-X base](https://github.com/electricsheephq/lcm-x/tree/efc32d53291fbd4955c5c7e63c40770a38b400ec)
- [upstream parser](https://github.com/NousResearch/hermes-agent/blob/6f7a7991bb069db07ae74a479823ce8310f8c7e0/pm/plugin_declarations.py#L110)
- [upstream installation/consent](https://github.com/NousResearch/hermes-agent/blob/6f7a7991bb069db07ae74a479823ce8310f8c7e0/hermes_cli/plugins_cmd_install.py#L84)
- [reference parser/installer](https://github.com/NousResearch/hermes-agent/blob/f97608f178d1ffeca59860195ab7da295f7c8e5f/hermes_cli/plugin_python_deps.py#L94)
- [reference enable](https://github.com/NousResearch/hermes-agent/blob/f97608f178d1ffeca59860195ab7da295f7c8e5f/hermes_cli/plugins_cmd.py#L1329)

## S1-C.4 qualification-only delta (4536f171)
Existing product consumer inventory above is unchanged. New harness consumers in dependency_qualification.py:305-344 read the synthetic-home scanner policy, call the exact PM update with interactive=False, preserve scanning-on refusal, restore policy in finally, scan the companion, and compare the existing state probe before recording b/e-core-version-no-rebuild. No manifest/dependency range/default/installer or runtime consumer changed. Exact host pins remain PM6f7a7991bb069db07ae74a479823ce8310f8c7e0 and referencef97608f178d1ffeca59860195ab7da295f7c8e5f.
