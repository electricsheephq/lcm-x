# Two tiers, one board: fault-finding vs published claims (R3 §4 draft 1)

## The problem both extremes create
Measure only under publication discipline and you starve yourself of signal: pins, noise
floors, and pre-registration make every experiment slow, so you run few, and the faults in
your system stay unfound. Measure only at frontier speed and your numbers mean nothing:
undisclosed configs, moving datasets, and judge-of-the-day scoring produce boards where no
two rows are comparable — the current state of agent-memory leaderboards.

## The split
**Tier F — fault-finding.** Frontier readers and judges, current stacks, relaxed instrument
bar, speed over purity. The success metric is faults FOUND per unit time, not score. Our
LoCoMo lane is the reference case: one Tier-F run surfaced five harness bugs, a documented
corrupted-gold exposure, two config-class retrieval defects, and one genuine product
weakness (adversarial speaker attribution) — none of which our own green test suites had
caught. The 47% score was the least interesting output of that run.

**Tier P — published claims.** Full pins, the seven-point disclosure, A/A′ noise floors,
pre-registered gates, author≠judge adjudication. Slower by design, run less often, and the
only tier from which numbers leave the building.

Both tiers publish to the same board with a tier column — a disclosed fault-finding number
outranks an undisclosed marketing number.

## Why frontier-first for Tier F
The market's published baselines largely measure a world that no longer exists: fixed
readers frozen years back, configs tuned for models nobody deploys. A memory system is
consumed by CURRENT frontier agents; faults that matter are faults in that experience.
Testing against stale stacks optimizes for reproducing history. (Fixed weak readers retain
one legitimate role, which is why our V2-static rows still exist: they isolate
retrieval+delivery from answering ability — an instrument choice, disclosed as such.)

## The judge rule that spans both tiers
No model family grades its own homework: judge and answerer come from different families,
and judge identity + full prompts publish with every number. This is the same principle as
cross-model code review, applied to scoring — same-family judging carries a measured
leniency bias, and it is exactly the kind of silent config choice the disclosure standard
exists to surface.
