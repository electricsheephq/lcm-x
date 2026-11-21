// lcmx-eve import: copy the conversations of an agent's earlier Eve sessions into its LCM-X store.
//
// For an agent that ran on Eve before LCM-X was installed. It reads each session's saved stream from the agent's own
// Eve server (eve/client `snapshot()`), takes its text conversation (`transcriptOf`: user messages and completed
// replies; no tool calls, reasoning or attachments), and writes it to the store's spool as its own session
// `import-<eve session id>`. The engine host drains the spool: the running Eve process's host within seconds, or
// `lcmx-eve drain` when Eve is stopped. Idempotent: a second run stores nothing new for a session whose conversation
// did not grow. Each message keeps its original time (Eve's event time of the delivery or the completed reply); for
// sessions an earlier import stored with the import time, `retime` sets the original times afterwards.
import { spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import { existsSync, readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";

import { Client, transcriptOf } from "eve/client";

import { sha256 } from "./canonical.js";
import { RUNTIME_FILE } from "./constants.js";
import { createLcmxMemory } from "./index.js";

export interface ImportOptions {
  /** The agent's LCM-X root (as in defineLcmxMemory). */
  root: string;
  /** The agent's Eve server, for example http://127.0.0.1:8402. */
  baseUrl: string;
  /** Bearer for the Eve HTTP channel. */
  bearer: string;
  /** The memory slot's namespace, as defineLcmxMemory sets it: `lcmx:<agent>`. */
  namespace: string;
  /** The memory slot's scope value. Default `agent` (defineLcmxMemory). */
  scope?: string;
  /** The memory slot name (the file name under agent/memory/). Default `lcmx`. */
  slot?: string;
  /** Eve session ids to import. */
  sessions: readonly string[];
  /** After the import, set the original times on rows an earlier import stored (default true). Needs a drained spool. */
  retime?: boolean;
}

export interface ImportReport {
  session: string;
  messages: number;
  first?: string;
  last?: string;
  status: "imported" | "empty" | "failed";
  error?: string;
}

const uint32 = (value: number) => {
  const buffer = Buffer.allocUnsafe(4);
  buffer.writeUInt32BE(value);
  return buffer;
};
const prefixed = (buffer: Buffer) => Buffer.concat([uint32(buffer.byteLength), buffer]);
const scalar = (kind: string, value: string) => Buffer.concat([Buffer.from(`${kind}-v1\0`), prefixed(Buffer.from(value, "utf8"))]);
const hashed = (prefix: string, data: Buffer) => `${prefix}${createHash("sha256").update(data).digest("base64url")}`;

/** Eve 1.0.0-rc.6's memory scope key for a string scope (eve's createMemoryScope). */
export function eveScopeKey(namespace: string, scope: string): string {
  const ns = hashed("memns1_", scalar("namespace", namespace));
  const value = hashed("memscope1_", scalar("scope-scalar", scope));
  return hashed(
    "memscope1_",
    Buffer.concat([Buffer.from("eve-memory-composite-v1\0"), prefixed(Buffer.from(ns)), prefixed(Buffer.from(value))]),
  );
}

interface StreamEvent {
  type: string;
  data?: Record<string, unknown>;
  meta?: { at?: string };
}

/**
 * The original time of each transcript message: the time Eve consumed the delivery that carried a user message, or
 * completed the reply part of an assistant message. Messages are matched in order by role and text; a message with no
 * match (imported history) takes the time of the message before it, or the session's start.
 */
export function messageTimes(events: readonly StreamEvent[], messages: readonly { role: string; text: string }[]): string[] {
  const candidates: { role: string; text: string; at: string }[] = [];
  let start = "";
  for (const event of events) {
    const at = event.meta?.at;
    if (!at) continue;
    if (!start) start = at;
    if (event.type === "delivery.consumed") {
      const parts = (event.data?.parts as { kind?: string; text?: string }[] | undefined) ?? [];
      const text = parts.filter((part) => part.kind === "text" && typeof part.text === "string").map((part) => part.text).join("\n");
      candidates.push({ role: "user", text: text.trim(), at });
    } else if (event.type === "content.completed" && event.data?.phase === "reply" && event.data?.kind === "text") {
      candidates.push({ role: "assistant", text: String(event.data.text ?? event.data.value ?? "").trim(), at });
    }
  }
  const head = (text: string) => text.trim().slice(0, 80);
  let next = 0;
  let last = start;
  return messages.map((message) => {
    for (let index = next; index < candidates.length; index += 1) {
      const candidate = candidates[index]!;
      if (candidate.role !== message.role) continue;
      if (candidate.text === message.text.trim() || (candidate.text && head(candidate.text) === head(message.text))) {
        next = index + 1;
        last = candidate.at;
        return last;
      }
    }
    return last;
  });
}

/** Session ids of an agent's Eve sessions, from its local workflow data (`runs/*.json`, eve's session workflow). */
export function eveSessionIds(workflowData: string, before?: Date): string[] {
  const runs = join(workflowData, "runs");
  if (!existsSync(runs)) return [];
  const ids: { id: string; at: string }[] = [];
  for (const name of readdirSync(runs).filter((file) => file.endsWith(".json"))) {
    try {
      const run = JSON.parse(readFileSync(join(runs, name), "utf8")) as { runId?: string; workflowName?: string; createdAt?: string };
      if (!run.runId || run.workflowName !== "workflow//eve//workflowEntry") continue;
      if (before && run.createdAt && new Date(run.createdAt) >= before) continue;
      ids.push({ id: run.runId, at: run.createdAt ?? "" });
    } catch {
      // a run file being written: skip it
    }
  }
  return ids.sort((a, b) => a.at.localeCompare(b.at)).map((run) => run.id);
}

export interface ImportResult {
  reports: ImportReport[];
  /** The retime of sessions stored earlier, or why it did not run. */
  retime: RetimeReport | { error: string } | null;
}

export async function importEveSessions(options: ImportOptions): Promise<ImportResult> {
  const slot = options.slot ?? "lcmx";
  const scope = { key: eveScopeKey(options.namespace, options.scope ?? "agent"), namespace: options.namespace, value: options.scope ?? "agent" };
  // A store made by the live slot names its key: it must be the same, or the import would land in another store.
  const scopes = join(options.root, "scopes");
  for (const name of existsSync(scopes) ? readdirSync(scopes) : []) {
    try {
      const known = JSON.parse(readFileSync(join(scopes, name, "scope.json"), "utf8")) as { key: string; namespace: string };
      if (known.namespace === scope.namespace && known.key !== scope.key) {
        throw new Error(`scope key mismatch for ${scope.namespace}: store has ${known.key}, computed ${scope.key}`);
      }
    } catch (error) {
      if (error instanceof Error && error.message.startsWith("scope key mismatch")) throw error;
    }
  }
  const client = new Client({ baseUrl: options.baseUrl, auth: { bearer: options.bearer } });
  const memory = createLcmxMemory({ root: options.root, startWriter: false, recall: false });
  const reports: ImportReport[] = [];
  const times: Record<string, string[]> = {};
  try {
    for (const id of options.sessions) {
      try {
        const { events } = await client.sessions.attach(id).snapshot();
        const { messages } = transcriptOf(events);
        const history = messages.filter((message) => message.text.trim()).map((message) => ({ role: message.role, content: message.text }));
        if (!history.length) {
          reports.push({ session: id, messages: 0, status: "empty" });
          continue;
        }
        const at = messageTimes(events as unknown as StreamEvent[], messages.filter((message) => message.text.trim()));
        // One record per message, so each row keeps its own time. A store that already has the session holds the
        // positions it has and stores only what is new.
        for (let count = 1; count <= history.length; count += 1) {
          await memory.provider.capture!["turn.completed"]!({
            abortSignal: new AbortController().signal,
            messages: history.slice(0, count),
            operationId: `import:${id}:${count}`,
            memory: { scope, slot },
            session: { id: `import-${id}` },
            turn: null,
            recordedAt: at[count - 1] || undefined,
          } as never);
        }
        times[`import-${id}`] = at;
        reports.push({ session: id, messages: history.length, first: at[0], last: at[at.length - 1], status: "imported" });
      } catch (error) {
        reports.push({ session: id, messages: 0, status: "failed", error: error instanceof Error ? error.message.slice(0, 200) : String(error) });
      }
    }
  } finally {
    memory.close();
  }
  let retime: ImportResult["retime"] = null;
  if (options.retime !== false && Object.keys(times).length) {
    try {
      await drained(options.root, scope.key);
      retime = retimeStore(options.root, scope.key, times);
    } catch (error) {
      retime = { error: error instanceof Error ? error.message : String(error) };
    }
  }
  return { reports, retime };
}

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

/** Drains the store's spool (or waits while the running Eve's host drains it). The rows must exist before retime. */
async function drained(root: string, scopeKey: string, timeoutMs = 300_000): Promise<void> {
  const runtime = JSON.parse(readFileSync(join(root, RUNTIME_FILE), "utf8")) as { python: string; packageRoot: string; hostScript: string };
  const home = join(root, "scopes", sha256(scopeKey).slice(0, 24));
  const spool = join(home, "spool");
  const pending = () => (existsSync(spool) ? readdirSync(spool).filter((name) => !name.startsWith(".") && !name.endsWith(".failed")).length : 0);
  spawnSync(runtime.python, ["-I", runtime.hostScript, "drain", "--home", home, "--core", runtime.packageRoot], { encoding: "utf8" });
  const until = Date.now() + timeoutMs;
  while (pending() > 0) {
    if (Date.now() > until) throw new Error(`spool not drained after ${timeoutMs} ms (${pending()} records)`);
    await sleep(1_000);
  }
}

export interface RetimeReport {
  sessions: number;
  rows: number;
  skipped: Record<string, string>;
}

/**
 * Sets the original times on sessions that an earlier import stored with the import time. Run it after the spool is
 * drained (the rows must exist). Changes only `timestamp` of the `import-*` sessions' rows.
 */
export function retimeStore(root: string, scopeKey: string, times: Record<string, string[]>): RetimeReport {
  const runtime = JSON.parse(readFileSync(join(root, RUNTIME_FILE), "utf8")) as { python: string; packageRoot: string; hostScript: string };
  const home = join(root, "scopes", sha256(scopeKey).slice(0, 24));
  const result = spawnSync(runtime.python, ["-I", runtime.hostScript, "retime", "--home", home, "--core", runtime.packageRoot], {
    input: JSON.stringify({ sessions: times }),
    encoding: "utf8",
  });
  if (result.status !== 0) throw new Error(`retime failed: ${(result.stderr || result.stdout).slice(-300)}`);
  return JSON.parse(result.stdout.trim().split("\n").pop()!) as RetimeReport;
}
