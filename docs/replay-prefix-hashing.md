# Legacy replay snapshot prefix hashing

The reconciliation readers of version-1 snapshot proofs must keep their identity
bytes even after the native-recovery writer has retired. See
[the identity ledger](identity-ledger.md), especially the legacy-reader and
rollback-compatibility rules.

## What is optimized (#1090)

Both the store-tail matcher and the empty-session native-recovery reader search
candidate cuts longest first. They previously rebuilt replay identities and
serialized the entire candidate prefix at each cut. The store-tail matcher did
this even when the native proof namespace was empty. Engine-assembled and
session-end snapshot checks repeated the same serialization on eligible prefixes.

`ReconcileMixin._replay_snapshot_prefix_digests` builds a call-local prefix index
for the nonempty proof namespaces. Each row's full snapshot replay identity is
resolved and JSON-serialized once. Do not substitute occurrence-projected,
filtered, sanitized-tail or stored-row identities: the persisted proof fingerprints
the original full snapshot using `_message_replay_identity`'s default options.

The JSON settings are unchanged: `ensure_ascii=False`, `sort_keys=True`, and
`separators=(",", ":")`. Sorted envelope keys give exactly:

    {"messages":[<identity>,<identity>,...],"version":1}

One SHA-256 state starts with `{"messages":[`. Append each serialized identity,
separated by commas. At a candidate cut, copy the hash state and append only
`],"version":1}`. That copy hashes the exact legacy payload, without rehashing
previous identities or retaining their serialized byte buffers. A rolling digest
of digests would not be compatible and must not replace this procedure.

## Eligibility and ordering are unchanged

- Native recovery: nonempty prefix; no lossy-redacted replay identity anywhere
  in that prefix (including tool-call fields).
- Engine-assembled snapshot: both the LCM system-note anchor and generated-summary
  anchor must appear in the prefix.
- Session-end snapshot: generated-summary anchor only; its namespace is consumed
  solely when the caller explicitly permits session-end proof.
- All valid cuts remain eligible. Existing opaque durable digests do not record
  cardinality, so cutting the search off based on length would skip valid proof.
- The descending reconciliation decision order, exact native match, tail matching,
  folded mapping, scaffold handling and duplicate-over-loss fallback are unchanged.
- Empty proof namespaces do no snapshot hashing or payload resolution.

The index is not persisted or shared between calls. A later call resolves changed
externalized payloads again. No new metadata, schema, dependency, config key,
background job or Hermes host capability is introduced. Single-snapshot digest
readers/writers retain their existing version-1 serialization.

## Cost and proof boundary

For `n` messages with `B` total serialized identity bytes, snapshot indexing does
`n` identity resolutions, `n` serializations, and hashes `B + O(n)` bytes, using
`O(n)` digest entries plus the current row's identity/serialization. Anchor scans
inspect raw content once per row. This does **not** make all reconciliation linear:
other matcher predicates still scan, filter and slice candidate prefixes and can
remain quadratic. A longest/full-prefix match can also pay a modest constant-factor
indexing overhead compared with one legacy whole-snapshot digest.

`tests/test_issue_1090_replay_prefix_scaling.py` pins every-cut equality against the
unchanged single-snapshot readers, actual serialized bytes, namespace gates,
positive/rejected proof, lossy identities, Unicode, tool fields, externalized
media/result payloads, call-local freshness, identity counts, serialization counts
and hash-byte counts. Existing lifecycle/replay tests remain required; the prefix
helper alone is not proof of end-to-end compaction latency or production adoption.
