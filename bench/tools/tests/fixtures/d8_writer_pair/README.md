# Writer-produced D8 identity fixtures

Generated with [memorybench PR #17](https://github.com/electricsheephq/memorybench-benchmark-tool/pull/17), head `203ba758e9ca788179bfb47658a6c0f09a4af99a`, targeting `bench/locomo-fulltext-arm`.
From that writer checkout, with `LCMX` set to the consumer checkout:
```sh
bun scripts/emit-identity-fixture.ts "$LCMX/bench/tools/tests/fixtures/d8_writer_pair/a" A
bun scripts/emit-identity-fixture.ts "$LCMX/bench/tools/tests/fixtures/d8_writer_pair/b" B
bun scripts/emit-identity-fixture.ts "$LCMX/bench/tools/tests/fixtures/d8_writer_pair/c" A
```
Only one synthetic dataset byte differs in b; a and c use identical inputs. Real checkpoint/report writers, stub answers/verdicts, no model calls or corpus. The same assertions fail on output from the old writer at `a36c5cbe208c67cdcfd2baad7a17382a9f21dd42`: identities are missing on both sides and the changed byte is accepted. These subsets do not certify a real D4 pair or full D8 row.
