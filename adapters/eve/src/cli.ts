#!/usr/bin/env node
// lcmx-eve: set up and operate the LCM-X runtime of one Eve agent.
//
//   lcmx-eve setup  --root <dir> [--python <abs>] [--fastembed]   find or install Python, write <root>/runtime.json
//   lcmx-eve check  --root <dir>                                  engine loads, SQLite, FastEmbed, LLM key present
//   lcmx-eve drain  --root <dir>                                  drain every store's spool once (for a timer)
//   lcmx-eve audit  --root <dir>                                  lossless audit of every store
//
// The CLI runs from the installed package, so it knows the vendored core and the host script by its own path.
import { execFileSync, spawnSync } from "node:child_process";
import { existsSync, mkdirSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, isAbsolute, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { RUNTIME_FILE } from "./constants.js";

const PYTHON_VERSION = "3.12.16"; // uv-managed CPython; its SQLite is new enough (3.51.3+)
const MIN_SQLITE = [3, 51, 3];
const pkg = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const core = existsSync(join(pkg, "core", "store.py")) ? join(pkg, "core") : resolve(pkg, "..", ".."); // package, or a checkout
const hostScript = join(pkg, "host", "lcmx_eve_host.py");
const fastembedLock = join(pkg, "host", "requirements-fastembed.txt");

function fail(message: string): never {
  process.stderr.write(`lcmx-eve: ${message}\n`);
  process.exit(2);
}

function option(args: string[], name: string): string | undefined {
  const index = args.indexOf(name);
  return index >= 0 ? args[index + 1] : undefined;
}

function probe(python: string): { version: string; sqlite: string } | null {
  const result = spawnSync(python, ["-I", "-c", "import sys, sqlite3; print(sys.version.split()[0], sqlite3.sqlite_version)"], {
    encoding: "utf8",
  });
  if (result.status !== 0) return null;
  const [version, sqlite] = result.stdout.trim().split(" ");
  return version && sqlite ? { version, sqlite } : null;
}

function atLeast(version: string, minimum: number[]): boolean {
  const parts = version.split(".").map((part) => Number.parseInt(part, 10));
  for (let index = 0; index < minimum.length; index += 1) {
    const value = parts[index] ?? 0;
    if (value !== minimum[index]) return value > minimum[index]!;
  }
  return true;
}

function suitable(python: string): boolean {
  const found = probe(python);
  return Boolean(found && atLeast(found.version, [3, 11]) && atLeast(found.sqlite, MIN_SQLITE));
}

function which(command: string): string | null {
  const result = spawnSync("sh", ["-c", `command -v ${command}`], { encoding: "utf8" });
  const path = result.stdout.trim();
  return result.status === 0 && isAbsolute(path) ? path : null;
}

function findPython(explicit: string | undefined): string {
  if (explicit) {
    if (!isAbsolute(explicit)) fail("--python must be an absolute path");
    if (!suitable(explicit)) fail(`${explicit} is not Python 3.11+ with SQLite ${MIN_SQLITE.join(".")} or newer`);
    return explicit;
  }
  const uv = which("uv");
  if (uv) {
    let found = spawnSync(uv, ["python", "find", PYTHON_VERSION], { encoding: "utf8" });
    if (found.status !== 0) {
      process.stdout.write(`installing CPython ${PYTHON_VERSION} with uv\n`);
      execFileSync(uv, ["python", "install", PYTHON_VERSION], { stdio: "inherit" });
      found = spawnSync(uv, ["python", "find", PYTHON_VERSION], { encoding: "utf8" });
    }
    const path = found.stdout.trim();
    if (found.status === 0 && suitable(path)) return path;
  }
  for (const name of ["python3.14", "python3.13", "python3.12", "python3.11", "python3"]) {
    const path = which(name);
    if (path && suitable(path)) return path;
  }
  fail(
    `no Python 3.11+ with SQLite ${MIN_SQLITE.join(".")}+ found. Install uv (it installs CPython ${PYTHON_VERSION}) ` +
      "or pass --python <absolute path>.",
  );
}

function rootOf(args: string[]): string {
  const root = option(args, "--root") ?? process.env.LCMX_EVE_ROOT;
  if (!root || !isAbsolute(root)) fail("--root <absolute folder> (or LCMX_EVE_ROOT) is required");
  return root;
}

function runtime(root: string): { python: string; packageRoot: string; hostScript: string } {
  const file = join(root, RUNTIME_FILE);
  if (!existsSync(file)) fail(`no ${file}: run lcmx-eve setup --root ${root}`);
  return JSON.parse(readFileSync(file, "utf8"));
}

function homes(root: string): string[] {
  const scopes = join(root, "scopes");
  return existsSync(scopes) ? readdirSync(scopes).map((name) => join(scopes, name)) : [];
}

function host(python: string, args: string[]): { status: number; out: unknown } {
  const result = spawnSync(python, ["-I", hostScript, ...args, "--core", core], { encoding: "utf8" });
  let out: unknown = result.stdout.trim();
  try {
    out = JSON.parse(String(out));
  } catch {
    // keep the text
  }
  if (result.status !== 0 && result.stderr) process.stderr.write(result.stderr);
  return { status: result.status ?? 1, out };
}

function setup(args: string[]): void {
  const root = rootOf(args);
  mkdirSync(root, { recursive: true, mode: 0o700 });
  let python = findPython(option(args, "--python"));
  if (args.includes("--fastembed")) {
    const venv = join(root, "venv");
    if (!existsSync(join(venv, "bin", "python"))) execFileSync(python, ["-m", "venv", venv], { stdio: "inherit" });
    python = join(venv, "bin", "python");
    execFileSync(python, ["-m", "pip", "install", "--quiet", "--require-virtualenv", "-r", fastembedLock], { stdio: "inherit" });
  }
  const checked = host(python, ["check"]);
  if (checked.status !== 0) fail(`the engine does not load with ${python}: ${JSON.stringify(checked.out)}`);
  const config = { python, packageRoot: core, hostScript, package: pkg, createdAt: new Date().toISOString() };
  writeFileSync(join(root, RUNTIME_FILE), JSON.stringify(config, null, 2) + "\n", { mode: 0o600 });
  process.stdout.write(`${JSON.stringify({ runtime: join(root, RUNTIME_FILE), ...(checked.out as object) })}\n`);
}

async function importCommand(args: string[]): Promise<void> {
  const { eveSessionIds, importEveSessions } = await import("./import.js");
  const root = rootOf(args);
  const url = option(args, "--eve-url");
  const bearerFile = option(args, "--bearer-file");
  const namespace = option(args, "--namespace");
  const workflowData = option(args, "--workflow-data");
  if (!url || !bearerFile || !namespace || !workflowData) {
    fail("usage: lcmx-eve import --root <dir> --eve-url <url> --bearer-file <file> --namespace lcmx:<agent> --workflow-data <dir> [--before <ISO time>] [--skip id,id]");
  }
  const before = option(args, "--before");
  const skip = new Set((option(args, "--skip") ?? "").split(",").filter(Boolean));
  const sessions = eveSessionIds(workflowData!, before ? new Date(before) : undefined).filter((id) => !skip.has(id));
  const reports = await importEveSessions({ root, baseUrl: url!, bearer: readFileSync(bearerFile!, "utf8").trim(), namespace: namespace!, sessions });
  for (const report of reports) process.stdout.write(`${JSON.stringify(report)}\n`);
  const count = (status: string) => reports.filter((report) => report.status === status).length;
  process.stdout.write(`${JSON.stringify({ sessions: reports.length, imported: count("imported"), empty: count("empty"), failed: count("failed"), messages: reports.reduce((sum, report) => sum + report.messages, 0) })}\n`);
}

function main(): void {
  const [command, ...args] = process.argv.slice(2);
  if (command === "setup") return setup(args);
  if (command === "import") {
    importCommand(args).catch((error) => fail(String(error)));
    return;
  }
  if (command === "check") {
    const { python } = runtime(rootOf(args));
    const result = host(python, ["check"]);
    process.stdout.write(`${JSON.stringify(result.out)}\n`);
    process.exit(result.status);
  }
  if (command === "drain" || command === "audit") {
    const root = rootOf(args);
    const { python } = runtime(root);
    let status = 0;
    for (const home of homes(root)) {
      const result = host(python, [command, "--home", home]);
      process.stdout.write(`${JSON.stringify({ home, ...(result.out as object) })}\n`);
      status = Math.max(status, result.status);
    }
    process.exit(status);
  }
  process.stdout.write(
    "usage: lcmx-eve setup|check|drain|audit|import --root <absolute folder> [--python <abs>] [--fastembed]\n",
  );
  process.exit(command ? 2 : 0);
}

main();
