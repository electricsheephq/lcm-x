# Outside Contributions

How maintainers handle pull requests from outside contributors and outside agents, and what our
own agents follow. Contributor-facing rules are in [CONTRIBUTING.md](../../CONTRIBUTING.md); merge
rules are in [AGENTS.md: Review And Merge](../../AGENTS.md#review-and-merge).

## Outside pull requests

- First response within 5 business days.
- Read the whole diff before approving CI for a first-time or fork contributor.
- Review at the exact head, by a reviewer that is not the author. For AI-written changes, the
  reviewer is a different model than the one that wrote the change. The review obligation itself
  is in [AGENTS.md: Review And Merge](../../AGENTS.md#review-and-merge).
- The sections of the [PR template](../../.github/PULL_REQUEST_TEMPLATE.md) are required.
- A maintainer may finish an outside PR:
  - by committing on top of it, when the author allows edits; or
  - on a new branch that keeps the contributor's commits, with a PR body that credits
    "Based on #N by @author". Authorship rules are in
    [AGENTS.md: Non-Negotiable Invariants](../../AGENTS.md#non-negotiable-invariants).
- Explain every requested change and every close.

## Outside AI agents and bots

An agent-written PR must:

- say in the PR body that the change is agent-written;
- link the issue it addresses;
- for a bug fix, include a test that fails before the change;
- stay draft until a maintainer reviews it.

Maintainers may close agent PRs that ignore the PR template or touch unrelated files, with a
comment that says why.

## Our own agents

Agents that maintainers run follow [AGENTS.md](../../AGENTS.md) and the documents in
`docs/maintainers/`. They never set milestones, and they never merge outside the
[land-pr](../../.agents/skills/land-pr/SKILL.md) skill.

## Security reports

Vulnerability details never go in a public issue or PR. Reporters use the private advisory path
in [SECURITY.md](../../SECURITY.md); when private reporting is unavailable, SECURITY.md allows a
minimal public issue that only asks for a private contact. If details arrive in a public issue or
PR, do not discuss them there; point the reporter to the private path.
