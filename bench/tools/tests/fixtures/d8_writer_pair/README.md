# Writer-produced D8 identity fixtures

Writer: [memorybench PR #17](https://github.com/electricsheephq/memorybench-benchmark-tool/pull/17), merged as [`802096d86782ebc8db240fbf0ebda39c3fd395c7`](https://github.com/electricsheephq/memorybench-benchmark-tool/commit/802096d86782ebc8db240fbf0ebda39c3fd395c7) on `bench/locomo-fulltext-arm`. Fixtures generated at the reviewed head [`a2bc93d46ae7681427d3af34b5e836cc6ae32016`](https://github.com/electricsheephq/memorybench-benchmark-tool/commit/a2bc93d46ae7681427d3af34b5e836cc6ae32016).
From that writer checkout, with `LCMX` set to the consumer checkout:
```sh
bun scripts/emit-identity-fixture.ts "$LCMX/bench/tools/tests/fixtures/d8_writer_pair/a" A
bun scripts/emit-identity-fixture.ts "$LCMX/bench/tools/tests/fixtures/d8_writer_pair/b" B
bun scripts/emit-identity-fixture.ts "$LCMX/bench/tools/tests/fixtures/d8_writer_pair/c" A
```
Only one synthetic dataset byte differs in b; a and c use identical inputs. Real checkpoint/report writers, stub answers/verdicts, no model calls or corpus. The same assertions fail on output from the old writer at `a36c5cbe208c67cdcfd2baad7a17382a9f21dd42`: identities are missing on both sides and the changed byte is accepted. These subsets do not certify a real D4 pair or full D8 row.
