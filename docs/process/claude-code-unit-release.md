# Releasing LCM-X for Claude Code

The rules are in `bench/specs/RELEASE-READINESS-V1.md`, "Claude Code unit releases". This is the
order of steps. The release manager decides, tags and publishes; agents prepare the candidate and
the gate's evidence.

1. **Candidate PR.** Set `version` in `adapters/claude-code/.claude-plugin/plugin.json` to the
   release version and add `.github/release-notes/claude-code-v<version>.md` (what it does, how to
   opt in, what it does not do yet, known issues). The notes do not name the core commit: the tag
   workflow adds it, with the core version, from `bundle.json`. It lands through `land-pr` like any
   PR. Its merge commit on `main` is the candidate. If the gate later fails, the fix PR's merge
   commit becomes the candidate.
2. **Build the candidate's asset:**
   `python scripts/build_claude_code_unit.py <candidate sha> --out <dir>`. Check its `.sha256` and
   that `bundle.json` names the candidate commit. (The tag's asset will differ from this one only
   in `bundle.json`'s `ref`, the tag name instead of the sha.)
3. **Run the gate on that asset** (the design's "Test plan with real Claude Code sessions"):
   - R1-R10 through Claude Code with the asset's plugin folder loaded by `--plugin-dir`, a scratch
     `HOME` and a scratch `CLAUDE_CONFIG_DIR`, each run recorded as the spec lists;
   - `<plugin folder>/lcm-x/scripts/lcm_x_capture.py audit --home <store home>` after each run and
     per session: exit 0 and `verified: true`;
   - the growth check when its inputs changed since its last run;
   - `python scripts/claude_code_unit_smoke.py --ref <candidate sha> --runtime venv`, then
     `--runtime found`, each on Python 3.11 and 3.14;
   - the install records (macOS, Linux).
4. **Receipt.** Write the result card (`RESULT-MANIFEST.json`, `RESULT.md`): counts, ids, hashes
   and timings, no transcript content, no machine-local paths. Report anything the spec sends to
   the release manager.
5. **Tag.** The release manager tags the candidate:
   `git tag -a claude-code-v<version> <candidate sha> -m "LCM-X for Claude Code <version>"` and
   pushes the tag. `.github/workflows/claude-code-release.yml` checks the tag (annotated, on
   `main`'s first-parent history, the manifest version, the notes), builds the asset, runs the CLI
   smoke on Python 3.11 and 3.14, and leaves a draft release with the asset, `.sha256` and
   `bundle.json`; the release body is the notes plus the bundled core line. If the run fails,
   nothing is published: fix through a new candidate and release the next patch version. The
   workflow only creates a release and never edits one: if any release for the tag exists, draft or
   published, it stops. To rebuild a draft, delete the draft and re-run the workflow by hand with
   the tag.
6. **Publish.** The release manager attaches the gate's result card to the draft, reads the
   workflow run, and publishes the draft as a prerelease (below 1.0.0) with "Set as the latest
   release" unticked: `gh release edit claude-code-v<version> --draft=false --prerelease --latest=false`.
7. **After the release.** A follow-up PR may move the manifest to the next `-dev` version.
   `scripts/claude_code_unit_delta.py` then lists what the next release must re-gate.
