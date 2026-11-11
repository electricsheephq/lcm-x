// lcmx-eve import: copy the conversations of an agent's earlier Eve sessions into its LCM-X store.
//
// For an agent that ran on Eve before LCM-X was installed. It reads each session's saved stream from the agent's own
// Eve server (eve/client `snapshot()`), takes its text conversation (`transcriptOf`: user messages and completed
// replies; no tool calls, reasoning or attachments), and writes it to the store's spool as its own session
// `import-<eve session id>`. The engine host drains the spool: the running Eve process's host within seconds, or
// `lcmx-eve drain` when Eve is stopped. Idempotent: the operation id is fixed per session, so a second run stores
// nothing new for a session whose conversation did not grow.
import { createHash } from "node:crypto";
import { existsSync, readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";

import { Client, transcriptOf } from "eve/client";

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
}

export interface ImportReport {
  session: string;
  messages: number;
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

export async function importEveSessions(options: ImportOptions): Promise<ImportReport[]> {
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
        await memory.provider.capture!["turn.completed"]!({
          abortSignal: new AbortController().signal,
          messages: history,
          operationId: `import:${id}:${history.length}`,
          memory: { scope, slot },
          session: { id: `import-${id}` },
          turn: null,
        } as never);
        reports.push({ session: id, messages: history.length, status: "imported" });
      } catch (error) {
        reports.push({ session: id, messages: 0, status: "failed", error: error instanceof Error ? error.message.slice(0, 200) : String(error) });
      }
    }
  } finally {
    memory.close();
  }
  return reports;
}
