# Changelog fragments

Record each change in its own fragment file here instead of editing `CHANGELOG.md`, so concurrent PRs never
conflict on that file.

- **One file per PR:** `changelog.d/<slug>.md`. The slug is lowercase `a-z0-9-` and unique. Putting the issue
  number first is recommended, for example `2345-doctor-write-lock.md`.
- **Content:** one or more bullet lines exactly as they would appear under `## Unreleased` in `CHANGELOG.md`, with
  the usual prefixes (`- New:`, `- Fix:`, `- Removed:`, `- Deprecated:`, `- Changed:`). Indent continuation lines
  by two spaces. No headings.
- **Do not edit `CHANGELOG.md` in a normal PR.** The release manager folds the fragments in at each release cut.

## Release cut

Fold changelog fragments: `python3 scripts/changelog_assemble.py`, then commit the result in the release-prep commit.

`python3 scripts/changelog_assemble.py --check` validates the fragments and prints what would be added without
changing anything; `--keep` assembles without deleting the fragment files.
