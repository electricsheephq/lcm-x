| Probe | Base 91a03989 | Previous head 04dc9385 | Fixed 566c498c |
|---|---|---|---|
| d1b (both arrival shapes) | repaired; no effective false certificate | row 2 falsely certified | repaired; no effective false certificate |
| d2 (both revision-bump shapes) | repaired | row 2 falsely certified | repaired |
| d3 current-revision control | sound | sound | sound |
| g legacy wider scan | row 3 visible then embedded | row 3 hidden by false certificate | row 3 visible then embedded |
| B1 manual skip/withheld | 2 (gap-fill); row 4 visible | 1; row 4 incorrectly covered | 2 (gap-fill); row 4 visible |
| b_parent_absent | N/A: no independent low-band key | row 4 falsely certified | safe early return before yield (EXPECTED_NEGATIVE) |
