#!/usr/bin/env node
// Vendors the LCM-X core at one pinned upstream commit into core/ (for npm pack), with BUNDLE.json naming
// the commit. Same file filter as the upstream Claude Code unit build (scripts/build_claude_code_unit.py):
// the commit minus repository tooling, plus the reader entry script. Reads the commit, never the working tree.
//   node scripts/vendor-core.mjs [commit]
import { execFileSync } from "node:child_process";
import { mkdirSync, rmSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

export const CORE_COMMIT = "eafe8f314648e5cb786604a207fe3556a18667cd"; // main, 2026-11-11 (the base of this adapter)
const here = dirname(fileURLToPath(import.meta.url));
const pkg = join(here, "..");
const repo = join(pkg, "..", "..");
const NOT_CORE_DIRS = [".agents/", ".github/", "adapters/", "bench/", "benchmarking/", "benchmarks/", "companions/", "docs/", "release/", "tests/"];
const NOT_CORE_FILES = new Set([".gitignore", "AGENTS.md", "CLAUDE.md", "CODE_OF_CONDUCT.md", "CONTRIBUTING.md", "pytest.ini", "ruff.toml"]);
const ENTRY_SCRIPTS = new Set(["scripts/lcm_x_mcp.py"]);

const isCore = (path) =>
  path.startsWith("scripts/") ? ENTRY_SCRIPTS.has(path) : !NOT_CORE_DIRS.some((dir) => path.startsWith(dir)) && !NOT_CORE_FILES.has(path);

const ref = process.argv[2] ?? CORE_COMMIT;
const git = (...args) => execFileSync("git", ["-C", repo, ...args], { maxBuffer: 1 << 30 });
const commit = git("rev-parse", "--verify", `${ref}^{commit}`).toString().trim();
const files = git("ls-tree", "-r", "--name-only", commit).toString().split("\n").filter(Boolean).filter(isCore);
const out = join(pkg, "core");
rmSync(out, { recursive: true, force: true });
mkdirSync(out, { recursive: true });
execFileSync("tar", ["-x", "-C", out], { input: git("archive", "--format=tar", commit, "--", ...files), maxBuffer: 1 << 30 });
writeFileSync(join(out, "BUNDLE.json"), JSON.stringify({ upstream: "electricsheephq/lcm-x", commit, files: files.length }, null, 2) + "\n");
console.log(`vendored ${files.length} core files from ${commit.slice(0, 12)} into core/`);
