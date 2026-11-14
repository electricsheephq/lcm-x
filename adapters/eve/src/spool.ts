import { mkdir, open, rename, writeFile } from "node:fs/promises";
import { join } from "node:path";

import type { Json } from "./canonical.js";

export const API = "lcmx.eve.capture/v1";

export interface SpoolRecord {
  api: typeof API;
  op: string;
  phase: "turn.completed" | "compaction.requested";
  at: string;
  session: { id: string };
  turn: { id: string; sequence: number; input: string[] } | null;
  scope: { key: string; namespace: string; slot: string };
  base: { count: number; chain: string };
  total: number;
  messages: { d: string; m: Json }[];
  compaction?: { modelId: string; usageInputTokens: number | null };
}

let lastMs = 0;
let counter = 0;

/** A name that sorts in write order inside this process and, by time, across processes. */
function recordName(): string {
  lastMs = Math.max(lastMs, Date.now());
  counter = (counter + 1) % 1_000_000;
  return `${String(lastMs).padStart(13, "0")}-${process.pid}-${String(counter).padStart(6, "0")}.json`;
}

async function syncDir(path: string): Promise<void> {
  const handle = await open(path, "r");
  try {
    await handle.sync();
  } finally {
    await handle.close();
  }
}

/**
 * Write one record durably: a temporary file, fsync, rename into `spool/`, fsync the folder. The
 * writer only reads `*.json`, so it never sees a half-written record.
 */
export async function writeRecord(home: string, record: SpoolRecord): Promise<string> {
  const folder = join(home, "spool");
  await mkdir(folder, { recursive: true, mode: 0o700 });
  const name = recordName();
  const temporary = join(folder, `.${name}.tmp`);
  const handle = await open(temporary, "wx", 0o600);
  try {
    await handle.writeFile(JSON.stringify(record), "utf8");
    await handle.sync();
  } finally {
    await handle.close();
  }
  const target = join(folder, name);
  await rename(temporary, target);
  await syncDir(folder);
  return target;
}

/** One line of provider trouble in `<root>/logs/provider.jsonl`; never throws. */
export async function logProblem(root: string, event: string, detail: Record<string, unknown>): Promise<void> {
  try {
    const folder = join(root, "logs");
    await mkdir(folder, { recursive: true, mode: 0o700 });
    await writeFile(join(folder, "provider.jsonl"), JSON.stringify({ at: new Date().toISOString(), event, ...detail }) + "\n", {
      flag: "a",
      mode: 0o600,
    });
  } catch {
    // nothing left to report to
  }
}
