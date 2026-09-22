# Review evidence provenance

The `AI review exact-head` check treats a repository dispatch as a request to
inspect existing GitHub pull-request reviews. A dispatch may name two review
objects by `lane` and numeric `review_id`; it may not submit reviewer scores,
verdicts, findings, identities, or receipt bodies.

The protected-main workflow fetches each named review through GitHub's pull
request review API. The validator derives the stored receipt from that fetched
object:

- GitHub's review author ID and login determine reviewer identity and must match
  the lane's protected publisher binding.
- GitHub's review ID determines the stored receipt ID.
- GitHub's `commit_id` must equal the pull request head, and the review's
  submission time becomes the receipt issue time.
- The review author must publish the exact structured body below. That body
  supplies the repository, pull request, base and head SHAs, lane, task ID,
  verdict, integer score, integer finding count, evidence digest, expiry, and
  policy version. Values are validated without score conversion.

```text
<!-- lcm-x-ai-review:v1
{"base_sha":"<40 lowercase hex>","evidence_digest":"<64 lowercase hex>","expires_at":"<timezone timestamp>","findings":0,"head_sha":"<40 lowercase hex>","lane":"acceptance|adversarial","policy_version":"1","pr_number":123,"repository":"electricsheephq/lcm-x","schema_version":"1","score":97,"task_id":"<safe opaque id>","verdict":"PASS"}
-->
```

The complete review body must be this marker and exact-field JSON object. Missing,
extra, malformed, stale, mismatched, low-scoring, or unresolved evidence fails
closed. An ordinary GitHub `COMMENTED` or `APPROVED` review is insufficient. A
recognizable bot name is also insufficient.

Protected policy currently binds `acceptance` to GitHub account ID `298367747`
(`evaos-code-review-bot[bot]`) and `adversarial` to account ID `199175422`
(`chatgpt-codex-connector[bot]`). These are identity allowlist entries, not a
claim that either bot currently publishes the structured contract. No genuine
publisher artifact satisfying this format was available during implementation.
Until both publishers are independently configured and their fetched GitHub
reviews pass the contract, the gate is expected to remain red.

Successful packets store the normalized receipts and review references. Every
peer reconciliation and final target read fetches the referenced GitHub reviews
again and compares the packet with newly derived receipts. A producer-authored
packet cannot preserve itself. The two publicly withdrawn legacy receipt IDs
remain explicitly rejected, and repository dispatches cannot submit replacement
receipt IDs because review IDs must be positive integers fetched from GitHub.

This design relies on the protected workflow source and its GitHub App check
writer. Local fixtures use synthetic review objects and prove validator behavior
only. They do not prove publisher onboarding, a real dispatch, a successful
protected check, merge readiness, release readiness, or runtime readiness.
