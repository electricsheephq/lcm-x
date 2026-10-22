# #1822 evidence

Synthetic cost and correctness receipts for candidate158191079b4ec61e676f9d03ea3f34069543e8fb. No customer data.

Re-score the frozen4800 samples: `python score.py`. This checks all eight original acceptance limits and writes cost-summary.json.

To re-run in a separate sandbox, set LCMX_PYTHON, LCMX_BASE, LCMX_CANDIDATE, LCMX_PRE1628 and LCMX_HOST_STUB, then `python bench.py 200`. The public launcher replaces only machine path constants; timed worker and scorer are exact copies. Full original script hashes are in run-metadata.json; T2 retains the original scripts.

RESULT.md is the public result card text; run-metadata.json contains its public-safe metadata and verified T2 tag/asset hashes. Complete RESULT-MANIFEST.json and all original evidence are preserved at that tag. The explicit SPEC file boundary excludes a T1 scoreboard edit.
