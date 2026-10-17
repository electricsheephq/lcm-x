# Portable recall distribution dossier

Tracker: [#1010](https://github.com/electricsheephq/lcm-x/issues/1010).
This is a local preview and submission preparation record, not store acceptance.

## Artifacts

`python3.11 scripts/package_portable.py --output /absolute/new/artifacts`
builds three self-contained ZIPs from a clean committed candidate: Claude Code,
manual Codex, and hook-free MCP-only. Each records the source commit and file
hashes. The runtime uses the existing LCM-X code under its MIT attribution;
there are no required third-party runtime dependencies. Python 3.11+ is required.
The launcher detects an installed newer Python when macOS `python3` is older;
an explicit absolute Python command is recommended on other platforms.
Generated artifacts and local configuration never enter repository source.

Each package includes a configuration preparer that refuses existing output
and creates new reviewable fragments. It never merges into live host settings.
The operator chooses storage root, project namespace, host instance and approved
transcript root. Sessions use dedicated corpora. Installation and capability
qualification receipts must name the actual artifact checksum and host version.

## Capability claims

Hermes: existing native context engine; this preview does not alter that route.
Claude Code and Codex: native compaction remains host-owned. Until named native
capture/compact/injection/expansion receipts pass, qualify them as MCP-only with
explicit ingestion. Hook configuration, manifest validation, emulated hook
payloads or a successful MCP handshake do not establish native qualification.
No context replacement or universal lossless-compaction claim is permitted.
Installed-version observations and their exact limits are recorded in
[portable qualification](portable-qualification.md).

## Claude plugin bundle dossier

Route: [plugin submission](https://claude.com/docs/plugins/submit), distinct from
[connector submission](https://claude.com/docs/connectors/building/submission)
and the [official marketplace](https://code.claude.com/docs/en/plugins/publish).
The ZIP supplies `.claude-plugin/plugin.json`, bundled local MCP, hooks, a
source-linked recall skill, README and license. Run `claude plugin validate
--strict --json /absolute/extracted/package` and install that actual package.
The developer portal/account role, public listing metadata, publisher identity,
privacy/terms/support URLs, icon/screenshots, repository/path and commit-bound
review must be completed by the owner before submission. Do not invent those
fields or treat the existing source repository URL as a reviewed listing.
No submission is performed by this work.

## OpenAI local-support dossier

Route: [OpenAI submission](https://developers.openai.com/plugins/deploy/submission)
and [Claude plugin guide](https://developers.openai.com/plugins/guides/submit-claude-plugin).
The manual Codex ZIP includes hooks; it is ineligible for ordinary public
submission. The separate hook-free ZIP has one local stdio MCP server and recall
skill. It requires the documented OpenAI-contact route for local MCP support;
it is not a public HTTPS submission package. Hosting was not authorized and no
endpoint is fabricated. Contact/publisher eligibility and acceptance of a local
server remain owner gates. Public publication, workspace publication and local
installation are different actions.

For a later approved submission, prepare the exact review form, publisher and
organization eligibility, five positive and three negative tool cases, a demo
video, review credentials if the chosen product needs an account, data handling
and support details. Keep credentials outside the ZIP and only in approved
secret references. The existing synthetic qualification receipts can supply
review examples but do not imply directory approval.

## Data handling and rollback

Store locally in the explicitly chosen directory. Hooks read only the configured
transcript root and identified synthetic or user-authorized sessions. No capture
writes host transcripts. Recall results are supplied to the host's selected
model provider; local storage does not imply that the host model runs locally.
BGE is optional local inference. Jev requires separate explicit public/synthetic
permission; private egress is disabled by default. No retention promises are
invented for external vendors.

Disable the plugin or remove only its prepared hook/MCP entries; native host
operation remains available. Retain saved corpora and source transcripts.
This preparation changes no installed global settings, customer runtime,
release, service, billing or publication state.

## Prepared tool review cases

Use a new synthetic namespace and explicit session, ingest one fixture through
the operator CLI, then use the source handle returned by recall. These cases are
part of the actual-package installation smoke, not invented live credentials.

| Case | Request / expected result |
| --- | --- |
| Positive 1 | lcm_recall for the synthetic ticket in session A returns unchanged cited source text. |
| Positive 2 | lcm_expand of that returned handle reconstructs the identical supporting span. |
| Positive 3 | lcm_describe for A returns its scoped store/DAG overview. |
| Positive 4 | lcm_status for A reports baseline scoring, explicit ingestion and no native context replacement. |
| Positive 5 | A small-token lcm_expand page followed by its next_offset reconstructs the full span without changing characters. |
| Negative 1 | Expanding A's handle in B fails with reference_corpus_mismatch. |
| Negative 2 | Adding a filesystem-path argument to a recall tool fails with unsupported_argument. |
| Negative 3 | Calling an ingestion tool over MCP fails with unknown_tool_or_arguments; writes remain operator/adapter-only. |

A later review dossier must attach the actual receipt/artifact digests and a
vendor-appropriate demo. Listing metadata, privacy/terms/support URLs and publisher
eligibility remain owner-supplied fields; no placeholder becomes a public claim.
