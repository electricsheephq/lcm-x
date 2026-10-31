# Long real-host regression

`tests/test_real_long_session.py` runs 300 seeded turns through the real Hermes
`AIAgent.run_conversation` loop, with a cold process restart after turn 150.
The restart uses `SessionDB.get_messages_as_conversation(...,
repair_alternation=True)`, the reader used by ACP's session restore path.
The test skips without a prepared host, like the other real-host regressions.

Prepare a host and export `LCM_REAL_HERMES_PYTHON` and `LCM_REAL_HERMES_SRC`
from the resulting JSON, using the same preparation as `real-host.yml`:

```sh
python -m bench.instruments.reliability.ci prep --host ref-0.21.5 --root "$RUNNER_TEMP/hosts" --out "$RUNNER_TEMP/hosts.json"
nice -n 19 python -m pytest -p no:cacheprovider -q tests/test_real_long_session.py
```

The reliability kit's fake provider serves localhost HTTP; its socket guard
blocks external network access. Replies and summaries are deterministic, with
no live model. The summary callback additionally preserves the objective when
it appears in its input, including at condensation. Retrieval uses FTS with
embeddings and reranking disabled. Real `read_file` calls supply single and
parallel groups, with large results stored through LCM externalization.

Hermes enforces a 64K startup floor. As in the other probes, initialization uses
that floor and `engine.update_model` sets the test's 32K window afterwards.
The fresh tail is eight rows; leaf chunks are 2K tokens. The test requires at
least five successful compactions and at least one non-leaf summary.

The lane's assertion labels deliberately follow its specification. Their
mapping to `docs/invariants.md` is:

| Lane label | Check | Catalogue |
|---|---|---|
| I1 | Exactly-once wire-message occurrences | I1 |
| I2 | Window, newest user, objective, pairs, verbatim fresh tail | I3, I4, I8, I9 |
| I3 | Grep, recall and verbatim expansion of aged facts and payloads | I2 |
| I4 | Nonempty summaries, valid sources, coverage exactly through the published frontier, uncovered rows still live | I14 |
| I5 | Stable stored IDs and no replay or skipped rows after restart | I1 |

Checkpoints run every 25 turns. Summary-source coverage must match the
published frontier exactly: every stored row at or before the frontier is a
summary source, and every later row is still present in live history. Hosts
compact at different turns, so no per-host row counts are pinned; compaction
progress is enforced by the per-turn window check and the minimum compaction
count. Row digests include store IDs and
wire payloads, expand externalized data, and omit host stamps and clocks.
Only the isolated home prefix is normalized; user and result bytes are checked
against their independently recorded originals within each run.

Set `LCM_LONG_SESSION_ARTIFACT` to the desired JSON output filename. Its default
is in the pytest temporary directory. The JSON contains per-turn and
per-compaction wall times plus checkpoint digests/counts. Timings are recorded,
never asserted. Each of the two host processes has a 175-second safety timeout.
The pytest temporary directory also retains provider logs, host logs, the
independent message ledger, and the isolated stores for diagnosis.
