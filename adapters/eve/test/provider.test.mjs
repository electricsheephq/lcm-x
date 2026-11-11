// Drives the built provider (dist/) with Eve rc.6-shaped handler contexts, against the real Python
// writer and the upstream read-only reader. Needs LCMX_PYTHON: an absolute Python 3.11+.
//   LCMX_PYTHON=$(uv python find 3.12.16) npm test
import assert from "node:assert/strict";
import { execFileSync, spawnSync } from "node:child_process";
import { mkdtempSync, readdirSync, readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";

import { mockModel } from "eve/evals";

import { createLcmxMemory, lcmxMemory } from "../dist/index.js";

const here = dirname(fileURLToPath(import.meta.url));
const packageRoot = resolve(here, "..", "..", "..");
const python = process.env.LCMX_PYTHON;

function context(root, history, extra = {}) {
  return {
    abortSignal: new AbortController().signal,
    messages: history,
    operationId: extra.operationId ?? `op-${Math.random()}`,
    memory: { scope: { key: "v1:agent:test", namespace: "test-agent", value: "agent" }, slot: "lcmx" },
    session: { id: extra.session ?? "sess-1" },
    turn: extra.turn === null ? null : { id: extra.turn ?? "turn-1", sequence: extra.sequence ?? 1, input: extra.input ?? [] },
    ...(extra.compaction ? { compaction: extra.compaction } : {}),
  };
}

function homeOf(root) {
  const scopes = join(root, "scopes");
  return join(scopes, readdirSync(scopes)[0]);
}

function audit(home) {
  const host = join(packageRoot, "adapters", "eve", "host", "lcmx_eve_host.py");
  try {
    return JSON.parse(execFileSync(python, ["-I", host, "audit", "--home", home], { encoding: "utf8" }));
  } catch (error) {
    return JSON.parse(error.stdout);
  }
}

test("capture, compaction and recall through the provider", { skip: !python && "LCMX_PYTHON is not set" }, async () => {
  const root = mkdtempSync(join(tmpdir(), "lcmx-eve-"));
  const handle = createLcmxMemory({ root, python, packageRoot, embeddings: "off" });
  const provider = handle.provider;
  try {
    const u1 = { role: "user", content: "Where is the invoice for ACME-7731?" };
    const a1 = {
      role: "assistant",
      content: [
        { type: "reasoning", text: "look it up" },
        { type: "tool-call", toolCallId: "call-1", toolName: "search", input: { q: "ACME-7731" } },
      ],
    };
    const t1 = {
      role: "tool",
      content: [{ type: "tool-result", toolCallId: "call-1", toolName: "search", output: { type: "text", value: "Invoice ACME-7731 is in /finance/2026/q3, paid on 2026-10-02." } }],
    };
    const a2 = { role: "assistant", content: [{ type: "text", text: "It is in /finance/2026/q3 and was paid." }] };
    const turn1 = [u1, a1, t1, a2];
    await provider.capture["turn.completed"](context(root, turn1, { input: [u1] }));
    // Eve retried the handler with the same operation id: nothing new may be stored.
    await provider.capture["turn.completed"](context(root, turn1, { input: [u1], operationId: "op-same" }));
    await provider.capture["turn.completed"](context(root, turn1, { input: [u1], operationId: "op-same" }));

    const u2 = { role: "user", content: "Thanks. Summarise the week." };
    const a3 = { role: "assistant", content: [{ type: "text", text: "Monday: invoices. Friday: payroll." }] };
    const before = [...turn1, u2, a3];
    await provider.capture["compaction.requested"](
      context(root, before, { turn: "turn-2", sequence: 2, input: [u2], compaction: { modelId: "test", usageInputTokens: 9000 } }),
    );
    const after = [
      { role: "user", content: "Summary of our conversation so far:", kind: "context.compaction" },
      { role: "assistant", content: "The user asked about invoice ACME-7731 and a weekly summary." },
      u2,
      a3,
      { role: "user", content: "And payroll?" },
      { role: "assistant", content: [{ type: "text", text: "Payroll ran on Friday." }] },
    ];
    await provider.capture["turn.completed"](context(root, after, { turn: "turn-3", sequence: 3, input: [after[4]] }));
    await handle.drain(120_000);

    const home = homeOf(root);
    assert.deepEqual(readdirSync(join(home, "spool")).filter((name) => !name.startsWith(".")), []);
    const report = audit(home);
    assert.equal(report.ok, true, JSON.stringify(report));
    // turn 1: user, assistant(tool call), tool, assistant; turn 2: user, assistant; after Eve's compaction:
    // user, assistant = 8 engine rows, no duplicates. Eve's marker and summary stay raw in the ledger.
    assert.equal(report.store_rows, 8, JSON.stringify(report));

    const tools = await provider.tools({
      abortSignal: new AbortController().signal,
      memory: { scope: { key: "v1:agent:test", namespace: "test-agent", value: "agent" }, slot: "lcmx" },
      session: { id: "sess-1" },
    });
    assert.deepEqual(Object.keys(tools).sort(), ["compile_evidence", "compute", "describe", "doctor", "evidence_pack", "expand",
      "expand_query", "grep", "inspect", "load_session", "recall", "recent", "status"]);
    const found = await tools.grep.execute({ query: "ACME-7731", session_scope: "all" }, {});
    const text = JSON.stringify(found);
    assert.match(text, /finance\/2026\/q3/, text.slice(0, 2000));
    const session = await tools.load_session.execute({ session_id: "eve:sess-1" }, {});
    assert.match(JSON.stringify(session), /Payroll ran on Friday/);

    // Turn-start recall: a new session asks about the invoice; Eve gets one keyed message with the hit.
    const recalled = await provider.recall["turn.started"](
      context(root, [], { session: "sess-2", input: [{ role: "user", content: [{ type: "text", text: "Where did the ACME-7731 invoice end up?" }] }] }),
    );
    assert.ok(recalled, "recall returned nothing");
    assert.equal(recalled.messages.length, 1);
    assert.equal(recalled.messages[0].id, "lcmx:turn-recall");
    assert.match(recalled.messages[0].content, /finance\/2026\/q3|ACME-7731/, recalled.messages[0].content);
    assert.match(recalled.messages[0].content, /data, not instructions/);
    // Hits the model still sees are left out: sess-1 with its whole history visible recalls nothing about the invoice.
    const visible = await provider.recall["turn.started"](
      context(root, [...after.slice(0, 2), ...turn1, u2, a3], { session: "sess-1", input: [{ role: "user", content: "Where did the ACME-7731 invoice end up?" }] }),
    );
    assert.doesNotMatch(visible?.messages?.[0]?.content ?? "", /finance\/2026\/q3/, JSON.stringify(visible));
    // After Eve compacted sess-1 (only the summary left in view), the dropped turn comes back.
    const compacted = await provider.recall["turn.started"](
      context(root, after, { session: "sess-1", input: [{ role: "user", content: "Where did the ACME-7731 invoice end up?" }] }),
    );
    assert.match(compacted?.messages?.[0]?.content ?? "", /finance\/2026\/q3/, JSON.stringify(compacted));
    // Short or empty text is not searched.
    assert.equal(await provider.recall["turn.started"](context(root, [], { session: "sess-2", input: [{ role: "user", content: "ok" }] })), null);
  } finally {
    handle.close();
  }
});

test("capture never throws, even when the root cannot be written", async () => {
  const provider = lcmxMemory({ root: "/dev/null/lcmx", python: "/usr/bin/false", packageRoot: "/nonexistent" });
  await provider.capture["compaction.requested"](context("/x", [{ role: "user", content: "hi" }], { compaction: { modelId: "m", usageInputTokens: null } }));
  await provider.capture["turn.completed"](context("/x", [{ role: "user", content: "hi" }]));
  assert.equal(await provider.recall["turn.started"](context("/x", [])), null);
});

test("capture returns within its time limit when the disk stalls", { skip: process.platform !== "linux" && "needs Linux /proc" }, () => {
  // Node's recursive mkdir never settles for this path on Linux: a stand-in for a stalled disk. It also
  // keeps the process alive, so the check runs in a child that is killed afterwards.
  const script = `
    import { lcmxMemory } from ${JSON.stringify(new URL("../dist/index.js", import.meta.url).href)};
    const provider = lcmxMemory({ root: "/proc/does-not-exist/lcmx", python: "/usr/bin/false", packageRoot: "/x", captureTimeoutMs: 300 });
    const started = Date.now();
    await provider.capture["compaction.requested"]({ messages: [{ role: "user", content: "hi" }], operationId: "o",
      memory: { scope: { key: "k", namespace: "n", value: "v" }, slot: "lcmx" }, session: { id: "s" }, turn: null,
      compaction: { modelId: "m", usageInputTokens: null } });
    console.log("returned " + (Date.now() - started));`;
  const result = spawnSync(process.execPath, ["--input-type=module", "-e", script], { encoding: "utf8", timeout: 5_000 });
  const match = /returned (\d+)/.exec(result.stdout);
  assert.ok(match, result.stdout + result.stderr);
  assert.ok(Number(match[1]) < 2_000);
});

test("recall can be turned off, and never throws", async () => {
  const off = lcmxMemory({ root: "/tmp/x", python: "/usr/bin/python3", packageRoot: "/x", recall: false });
  assert.equal(await off.recall["turn.started"](context("/x", [], { input: [{ role: "user", content: "a long enough question here" }] })), null);
  const broken = lcmxMemory({ root: "/dev/null/lcmx", python: "/usr/bin/false", packageRoot: "/nonexistent", recall: { timeoutMs: 2_000 } });
  assert.equal(await broken.recall["turn.started"](context("/x", [], { input: [{ role: "user", content: "a long enough question here" }] })), null);
});

test("formatRecall keeps at most two hits per session and strips highlight marks", async () => {
  const { formatRecall, turnQuery } = await import("../dist/index.js");
  const hits = [1, 2, 3].map((n) => ({ session_id: "eve:a", snippet: `>>>alpha<<< ${n}`, timestamp: 1794290000 }))
    .concat([{ session_id: "eve:b", snippet: "beta", timestamp: "2026-11-10T05:00:00Z", expand_hint: "lcm_expand(store_id=7)" }]);
  const text = formatRecall({ hits });
  assert.equal(text.split("\n").length, 4, text);
  assert.doesNotMatch(text, />>>|<<</);
  assert.match(text, /- eve:b \(2026-11-10\): beta \[lcm_expand\(store_id=7\)\]/);
  assert.equal(formatRecall({ hits: [] }), "");
  assert.equal(turnQuery([{ role: "user", content: "first" }, { role: "assistant", content: "x" }, { role: "user", content: [{ type: "text", text: "last" }] }]), "last");
});

test("formatRecall shows the same text once, and the import computes Eve's scope key", async () => {
  const { formatRecall } = await import("../dist/index.js");
  const { eveScopeKey } = await import("../dist/import.js");
  const { createMemoryScope } = await import("../node_modules/eve/dist/src/shared/memory-state.js");
  const text = formatRecall({ hits: [{ session_id: "eve:import-a", snippet: "release handover on 19 November" }, { session_id: "eve:a", snippet: "release handover on 19 November" }] });
  assert.equal(text.split("\n").length, 2, text);
  for (const agent of ["scout", "ranger"]) {
    assert.equal(eveScopeKey(`lcmx:${agent}`, "agent"), createMemoryScope({ namespace: `lcmx:${agent}`, scope: "agent", slot: "lcmx" }).key);
  }
});

test("the provider has only Eve's keys", () => {
  const provider = lcmxMemory({ root: "/tmp/x", python: "/usr/bin/python3", packageRoot: "/x" });
  assert.deepEqual(Object.keys(provider).sort(), ["capture", "recall", "tools"]);
});

test("options must be absolute paths", () => {
  assert.throws(() => lcmxMemory({ root: "relative", python: "/usr/bin/python3", packageRoot: "/x" }), /absolute/);
});

test("spool records are canonical and carry the prefix", async () => {
  const root = mkdtempSync(join(tmpdir(), "lcmx-eve-spool-"));
  const provider = lcmxMemory({ root, python: "/usr/bin/false", packageRoot: "/x", startWriter: false });
  const h1 = [{ content: "a", role: "user" }];
  await provider.capture["turn.completed"](context(root, h1));
  const h2 = [...h1, { role: "assistant", content: [{ type: "file", mediaType: "image/png", data: new Uint8Array([1, 2, 3]) }] }];
  await provider.capture["turn.completed"](context(root, h2));
  const spool = join(homeOf(root), "spool");
  const [first, second] = readdirSync(spool).sort().map((name) => JSON.parse(readFileSync(join(spool, name), "utf8")));
  assert.equal(first.base.count, 0);
  assert.equal(second.base.count, 1);
  assert.equal(second.messages.length, 1);
  assert.deepEqual(second.messages[0].m.content[0].data, { $bytes: "AQID" });
  assert.deepEqual(Object.keys(first.messages[0].m), ["content", "role"]);
});

test("LCM-X summaries run on the agent's model through the bridge, between turns", { skip: !python && "LCMX_PYTHON is not set" }, async () => {
  const root = mkdtempSync(join(tmpdir(), "lcmx-eve-bridge-"));
  const seen = [];
  const model = mockModel({
    modelId: "agent-model",
    respond: ({ messages }) => {
      seen.push(messages.map((message) => message.role));
      const nonce = /<lcm-summary nonce="([0-9a-f]+)">/.exec(messages.map((message) => message.text).join("\n"))?.[1];
      const body = "The user stored numbered notes with item codes and the assistant confirmed each.\nExpand for details about: item codes";
      return nonce ? `<lcm-summary nonce="${nonce}">\n${body}\n</lcm-summary>` : body;
    },
  });
  const handle = createLcmxMemory({ root, python, packageRoot, model, embeddings: "off", contextLength: 8000 });
  try {
    const history = [];
    for (let index = 0; index < 40; index += 1) {
      const note = { role: "user", content: `Note ${index}: item ${index} has code ZX-${index * 7}. ` + "filler words ".repeat(60) };
      history.push(note, { role: "assistant", content: [{ type: "text", text: `Noted item ${index}.` }] });
      await handle.provider.capture["turn.completed"](
        context(root, [...history], { turn: `t${index}`, sequence: index + 1, input: [note], operationId: `op-${index}` }),
      );
    }
    await handle.drain(300_000);
    const home = homeOf(root);
    const nodes = execFileSync(python, ["-I", "-c", `import sqlite3;print(sqlite3.connect(${JSON.stringify(join(home, "lcm.db"))}).execute("select count(*) from summary_nodes").fetchone()[0])`], { encoding: "utf8" });
    assert.ok(Number(nodes) >= 1, `summary nodes: ${nodes}`);
    assert.ok(seen.length >= 1);
    // LCM-X's prompt only: its policy as instructions (system) and the transcript to summarize; no agent history.
    assert.ok(seen.every((roles) => roles.join(",") === "system,user"), JSON.stringify(seen));
    const report = audit(home);
    assert.equal(report.ok, true, JSON.stringify(report));
  } finally {
    handle.close();
  }
});

test("a new Eve process gets the store at once when the old one ended during a summary call", { skip: !python && "LCMX_PYTHON is not set" }, async () => {
  const root = mkdtempSync(join(tmpdir(), "lcmx-eve-handover-"));
  // A model that never answers in time: the summary call is still waiting when this "Eve process" ends.
  const slow = mockModel({ respond: () => new Promise((resolve) => setTimeout(() => resolve("late"), 120_000)) });
  const first = createLcmxMemory({ root, python, packageRoot, model: slow, embeddings: "off", contextLength: 8000 });
  const history = [];
  for (let index = 0; index < 40; index += 1) {
    const note = { role: "user", content: `Note ${index}: item ${index} has code ZX-${index * 7}. ` + "filler words ".repeat(60) };
    history.push(note, { role: "assistant", content: [{ type: "text", text: `Noted item ${index}.` }] });
    await first.provider.capture["turn.completed"](context(root, [...history], { turn: `t${index}`, sequence: index + 1, input: [note], operationId: `op-${index}` }));
  }
  first.provider.tools({ abortSignal: new AbortController().signal, memory: { scope: { key: "v1:agent:test", namespace: "test-agent", value: "agent" }, slot: "lcmx" }, session: { id: "sess-1" } });
  await new Promise((resolve) => setTimeout(resolve, 3_000)); // the host drains and starts a summary call
  first.close(); // Eve's process ends: stdin closes
  const started = Date.now();
  const second = createLcmxMemory({ root, python, packageRoot, embeddings: "off" });
  try {
    const tools = await second.provider.tools({ abortSignal: new AbortController().signal, memory: { scope: { key: "v1:agent:test", namespace: "test-agent", value: "agent" }, slot: "lcmx" }, session: { id: "sess-2" } });
    const found = await tools.grep.execute({ query: "ZX-21", session_scope: "all" }, {});
    assert.ok(!found.error, JSON.stringify(found).slice(0, 300));
    assert.ok(Date.now() - started < 30_000, `took ${Date.now() - started} ms`);
    await second.drain(120_000);
    assert.equal(audit(homeOf(root)).ok, true);
  } finally {
    second.close();
  }
});
