# Review evidence provenance

The `AI review exact-head` check treats a repository dispatch as a request to
inspect existing GitHub pull-request reviews. A dispatch names review objects by
`lane` and numeric `review_id`; it cannot submit a verdict, findings, reviewer
identity, or assessment body.

The protected-main workflow fetches each named review through GitHub's pull
request review API. The validator derives the stored assessment from that fetched
object:

- GitHub's review author ID and login determine reviewer identity and must match
  the lane's protected publisher binding.
- GitHub's review ID identifies the original review.
- GitHub's `commit_id` must equal the pull request head, and the review's
  submission time becomes the assessment issue time.
- The review author must publish one terminal structured marker after any
  human-readable review summary. The marker supplies the repository, pull
  request, base and head SHAs, lane, explicit verdict, scope, the exact mapped
  named-risk list, string findings, string limitations, acceptance evidence,
  and policy version.

```text
Human-readable review summary.

<!-- lcm-x-ai-review:v2
{"acceptance_evidence":["<evidence>"],"base_sha":"<40 lowercase hex>","findings":[],"head_sha":"<40 lowercase hex>","lane":"acceptance|adversarial","limitations":[],"named_risks":["review-provenance-policy|lcm-memory-preservation"],"policy_version":"2","pr_number":123,"repository":"electricsheephq/lcm-x","schema_version":"2","scope":"<reviewed scope>","verdict":"PASS|BLOCKED|ABSTAIN"}
-->
```

The marker must occur exactly once and be the final content in the review body.
Missing, duplicate, extra-field, malformed, stale, mismatched, or unresolved
evidence fails closed. Success requires an explicit original `PASS` and an empty
`findings` list. GitHub `COMMENTED` or `APPROVED` state does not become `PASS`,
and a recognizable bot name is insufficient. Policy v1 numeric bodies and
producer-assigned scores are rejected.

Every PR requires one acceptance assessment. An adversarial assessment is also
required when protected source maps the changed paths to either named risk:
`review-provenance-policy` or `lcm-memory-preservation`. The mapping covers the
review workflow, validator, contributor/agent review rules and landing skill,
plus the LCM storage, compaction, lifecycle, and memory-preservation modules.
This is the protected automatic map, not a claim that other changes can never
need targeted adversarial review in their recorded PR acceptance. Labels and
dispatch claims cannot add or remove a mapped required lane.
Each original assessment must carry exactly the validator-computed named-risk
list, so an unrelated adversarial scope cannot satisfy a protected-path gate.

Protected policy currently binds `acceptance` to GitHub account ID `298367747`
(`evaos-code-review-bot[bot]`) and `adversarial` to account ID `199175422`
(`chatgpt-codex-connector[bot]`). These are identity allowlist entries, not a
claim that either bot currently publishes the structured contract. No genuine
protected LCM-X review artifact satisfying this v2 format was available during
implementation. Acceptance publisher support exists only as an unmerged
external candidate, while original adversarial publisher support remains
unavailable. Until the needed publishers are independently configured and their
fetched GitHub reviews pass this contract, affected gates are expected to remain
red. Tests cannot bootstrap a publisher or create genuine review evidence.

Successful packets store the normalized assessments, the entire original review
body, and review references. The workflow derives a SHA-256 tracking digest of
the entire body and a 24-hour tracking expiry from GitHub's submission time.
Those two fields are producer tracking metadata; they are not reviewer-authored
claims and never replace the original verdict. Every
peer reconciliation and final target read fetches the referenced GitHub reviews
again and compares the full packet with newly derived assessments. A producer-authored
packet cannot preserve itself. The two publicly withdrawn legacy receipt IDs
remain explicitly rejected, and repository dispatches cannot submit replacement
receipt IDs because review IDs must be positive integers fetched from GitHub.

This design relies on the protected workflow source and its GitHub App check
writer. A missing review locator is recorded on that target and does not revoke
independently valid peers. Local fixtures use synthetic review objects and prove
validator behavior only. They do not prove publisher onboarding, a real dispatch, a successful
protected check, merge readiness, release readiness, or runtime readiness.
