# LCM-X Maintainers

This is the entry point for people and agents who maintain `electricsheephq/lcm-x`. It indexes
the rules; it does not restate them. [AGENTS.md](AGENTS.md) stays the binding policy, and where
any document below disagrees with it, `AGENTS.md` wins.

## Who maintains

- `@100yenadmin` is the code owner ([.github/CODEOWNERS](.github/CODEOWNERS)) and the
  maintainer of record. Maintainer duties are listed in
  [AGENTS.md: Maintainer And Bot Roles](AGENTS.md#maintainer-and-bot-roles).

## Roles

| Role | May do | Governed by |
|---|---|---|
| Maintainers | Accept issues, set priority, make terminal dispositions, merge, release | [AGENTS.md](AGENTS.md) |
| Repository bots (CI, review bots) | Run checks and post review findings; their output is evidence | [AGENTS.md: Automation Boundary](AGENTS.md#automation-boundary) |
| AI agents run by maintainers | Triage, reproduce, implement, test, review, and prepare evidence; write only with a maintainer authorization | [AGENTS.md](AGENTS.md), the skills below |
| Outside contributors and outside agents | Open issues and pull requests | [CONTRIBUTING.md](CONTRIBUTING.md), [docs/maintainers/outside-contributions.md](docs/maintainers/outside-contributions.md) |

## Response targets

Targets, not guarantees. Business days.

| Item | First response |
|---|---|
| New issue | 3 business days |
| `P0` issue | Same day |
| Outside pull request | 5 business days |
| Security report | Private path per [SECURITY.md](SECURITY.md) |

## Index

| Document or tool | Use it for |
|---|---|
| [AGENTS.md](AGENTS.md) | Binding invariants, validation, review, merge, and release rules |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Contributor workflow and default checks |
| [SECURITY.md](SECURITY.md) | Private vulnerability reporting |
| [.agents/skills/triage-backlog/SKILL.md](.agents/skills/triage-backlog/SKILL.md) | Read-only triage of one issue, PR, or duplicate cluster |
| [.agents/skills/review-pr/SKILL.md](.agents/skills/review-pr/SKILL.md) | Read-only review of one PR at one exact head |
| [.agents/skills/land-pr/SKILL.md](.agents/skills/land-pr/SKILL.md) | Readiness checks and landing one PR |
| [.agents/skills/issue-sweep/SKILL.md](.agents/skills/issue-sweep/SKILL.md) | Bounded, maintainer-authorized sweep that puts triage onto GitHub |
| [docs/maintainers/label-policy.md](docs/maintainers/label-policy.md) | What every label means, who applies it, and when it comes off |
| [docs/maintainers/issue-triage.md](docs/maintainers/issue-triage.md) | Issue intake, comments, closing, post-merge, and stale sweeps |
| [docs/maintainers/outside-contributions.md](docs/maintainers/outside-contributions.md) | Outside PRs, outside agents, and our own agents |
| [docs/release-validation.md](docs/release-validation.md) | Offline pre-tag smoke lane (not the release gate) |
| [bench/specs/RELEASE-READINESS-V1.md](bench/specs/RELEASE-READINESS-V1.md) | The rc-first live release gauntlet |
| [scripts/maintainer/postmerge_issue_check.py](scripts/maintainer/postmerge_issue_check.py) | Read-only check for issues a merged PR said it fixes that are still open |
