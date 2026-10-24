# Independent semantic review receipt

Initial head147a94c4: BLOCKED for external symlink replacement guidance. Both exact
host sanitizers resolve the destination and reject an external symlink before
--force processing. Classified MERGE_BLOCKING, disposition fixed now.

Targeted delta d3a4153f454e796b34e5f669adf5cebd015897ef: PASS. Installer preserves
the link, tells the user to move only the link aside preserving its target,
then prints pinned native install without --force. Managed directories retain
explicit force guidance; fresh core-first/native-consent path unchanged.

Prior unchanged-product clearances retained: dependency-free core, stable carrier
range/no-op, actual native install consent ownership, no auto-force, exactSHApin,
legacy exception, missing-CLI linking fallback, operator remediation.

Later product delta d89b9883: doctor words restore explicit fastembed name and
warmup instruction required by existing tests; no consent/selection/runtime logic
changed. The two affected regression files pass (raw r2-ci-correction.log).

Boundary: source review only, no host installation, recall execution, profile
union or rebuild cost proof; qualification instruments not independently reviewed.
One initial pass and one targeted delta pass; one review-driven head change.
No public review comment/thread was posted, per the lane's no-comment boundary.
