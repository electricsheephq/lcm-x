// S1: lossless-claw REAL engine replay at 988dee8 (SPEC-TRACK-S §S1 + r3 §R2/§R5, r4 F1/N2). Wiring as in
// test/helpers.ts:259 createEngineWithDeps, but with the plugin's DEFAULT config (resolveLcmConfig), the z.ai
// CompleteFn, and a child tool loop behind deps.callGateway for lcm_expand_query delegation. No emulation.
import { DatabaseSync } from "node:sqlite";
import { createHash, randomUUID } from "node:crypto";
import { existsSync, mkdirSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { resolveLcmConfigWithDiagnostics } from "track-s-lossless/src/db/config.js";
import { closeLcmConnection, createLcmDatabaseConnection } from "track-s-lossless/src/db/connection.js";
import { LcmContextEngine } from "track-s-lossless/src/engine.js";
import { createLcmLogger } from "track-s-lossless/src/lcm-log.js";
import { estimateTokens } from "track-s-lossless/src/estimate-tokens.js";
import { FALLBACK_SUMMARY_MARKER } from "track-s-lossless/src/summary-fallback.js";
import { normalizeAgentId } from "track-s-lossless/src/plugin/openclaw-agent-ids.js";
import { createLcmGrepTool } from "track-s-lossless/src/tools/lcm-grep-tool.js";
import { createLcmDescribeTool } from "track-s-lossless/src/tools/lcm-describe-tool.js";
import { createLcmExpandTool } from "track-s-lossless/src/tools/lcm-expand-tool.js";
import { createLcmExpandQueryTool } from "track-s-lossless/src/tools/lcm-expand-query-tool.js";
import type { LcmConfig } from "track-s-lossless/src/db/config.js";
import type { LcmDependencies } from "track-s-lossless/src/types.js";
import { chat, callLog, clock, zaiComplete, ZAI_MODEL, type ChatMessage } from "./zai.js";

const TS = process.env.TRACK_S_OUT!, WT = process.argv[process.argv.indexOf("--worktree") + 1];
const argv = process.argv.slice(2);
const arg = (k: string, d?: string) => { const i = argv.indexOf(k); return i >= 0 ? argv[i + 1] : d; };
const SEED = arg("--seed")!, RUN = arg("--run", "1")!, SLICE = Number(arg("--slice", "0")), DRY = argv.includes("--dry-run");
const CADENCE_S = Number(arg("--cadence-s", "20")), BATCHES = Number(arg("--batches", "999"));
const OPEN_PROBES = (arg("--open-probes", "") || "").split(",").filter(Boolean);
// S6 A3: --prefix60k replays rows 0..the frozen prefix60k row (material/<seed>/prefix60k.json), then lowers the
// tokenBudget of that row's afterTurn so the engine's own threshold path fires there once; no probes.
const PREFIX = argv.includes("--prefix60k"), POP = PREFIX ? "prefix60k" : "full-stream";
// S7: --arm lossless-claw | lossless-claw-tuned (D7: default config + plugin config freshTailMaxTokens 64000); --open (D6:
// every batch answered with the public tools); --checkpoints r1,r2 (D1: material row indexes, probes from a store snapshot
// taken when that row has been fed; default = the last row). The budget is 272k for every S1 arm (D7).
const ARM = arg("--arm", "lossless-claw")!, OPEN = argv.includes("--open"), ARM_LABEL = ARM + (OPEN ? "-open" : "");
const ARM_PC: Record<string, unknown> = ({ "lossless-claw": {}, "lossless-claw-tuned": { freshTailMaxTokens: 64_000 } } as any)[ARM];
if (!ARM_PC) throw new Error(`unknown arm ${ARM}`);
const TOKEN_BUDGET = 272_000, GUARD = 20, READER_MAX_TOKENS = 8192;
const MAT = join(process.env.TRACK_S_MATERIAL!, SEED), RUN_DIR = join(TS, "lc-runs", SEED, RUN);
const jl = (p: string) => readFileSync(p, "utf8").split("\n").filter(Boolean).map((l) => JSON.parse(l));
const rd = (p: string) => JSON.parse(readFileSync(p, "utf8"));
const sha = (p: string) => createHash("sha256").update(readFileSync(p)).digest("hex");
const now = () => Date.now() - clock.t0;
const sleep = (ms: number) => new Promise((r) => setTimeout(r, Math.max(0, ms)));

// Verbatim copies of three private helpers of src/plugin/index.ts (:54 parseAgentSessionKey, :847, :862).
function parseAgentSessionKey(k: string) {
  const v = k.trim(); if (!v.startsWith("agent:")) return null;
  const p = v.split(":"); if (p.length < 3) return null;
  const agentId = p[1]?.trim(), suffix = p.slice(2).join(":").trim();
  return agentId && suffix ? { agentId, suffix } : null;
}
function buildSubagentSystemPrompt(p: { depth: number; maxDepth: number; taskSummary?: string }) {
  return ["You are a delegated sub-agent for LCM expansion.", `Depth: ${p.depth}/${p.maxDepth}`,
    "Return concise, factual results only.", p.taskSummary?.trim() || "Perform delegated LCM expansion work."].join("\n");
}
function textOf(c: unknown): string {
  if (typeof c === "string") return c;
  if (!Array.isArray(c)) return "";
  return c.map((b: any) => (typeof b === "string" ? b : b?.type === "text" ? b.text ?? "" : textOf(b?.content))).filter(Boolean).join("\n");
}
function readLatestAssistantReply(messages: unknown[]) {
  for (let i = messages.length - 1; i >= 0; i--) {
    const m = messages[i] as any; if (m?.role !== "assistant") continue;
    const t = textOf(m.content).trim(); if (t) return t;
  }
  return undefined;
}
// lossless-claw's normal recall guidance = the before_prompt_build prependSystemContext text (src/plugin/index.ts:345),
// read from the pinned source (the constant is not exported); its sha256 is recorded.
function recallPolicyPrompt(): string {
  const src = readFileSync(join(WT, "src/plugin/index.ts"), "utf8");
  const m = src.match(/const LOSSLESS_RECALL_POLICY_PROMPT = \[([\s\S]*?)\]\.join\("\\n"\);/);
  if (!m) throw new Error("recall policy constant not found at the pin");
  return (new Function(`return [${m[1]}]`)() as string[]).join("\n");
}

// ---- material -> OpenClaw AgentMessage shapes (genuine roles: user / assistant+toolCall / toolResult) ----
function toAgentMessage(r: any): any {
  const timestamp = Math.round(r.ts * 1000);
  if (r.role === "user") return { role: "user", content: r.content, timestamp };
  if (r.role === "tool") return { role: "toolResult", toolCallId: r.tool_call_id, toolName: "read",
    content: [{ type: "text", text: r.content }], isError: false, timestamp };
  const content: any[] = [{ type: "text", text: r.content }];
  if (r.tool_call_id) content.push({ type: "toolCall", id: r.tool_call_id, name: "read", arguments: { fixture: r.tool_call_id } });
  return { role: "assistant", content, timestamp };
}
function toChat(msgs: any[]): ChatMessage[] {
  return msgs.map((m) => {
    if (m.role === "toolResult" || m.role === "tool") return { role: "tool", tool_call_id: m.toolCallId, content: textOf(m.content) };
    if (m.role === "assistant") {
      const calls = Array.isArray(m.content) ? m.content.filter((b: any) => b?.type === "toolCall") : [];
      return { role: "assistant", content: textOf(m.content) || null, ...(calls.length ? { tool_calls: calls.map((b: any) => ({
        id: b.id, type: "function", function: { name: b.name, arguments: JSON.stringify(b.arguments ?? b.input ?? {}) } })) } : {}) };
    }
    return { role: m.role === "system" ? "system" : "user", content: textOf(m.content) };
  });
}
const flatten = (msgs: ChatMessage[]) => msgs.map((m) => `[${m.role}${m.tool_call_id ? ` ${m.tool_call_id}` : ""}]\n${m.content ?? ""}` +
  (m.tool_calls ? `\n(tool calls: ${JSON.stringify(m.tool_calls)})` : "")).join("\n\n");

// ---- engine wiring ----
function makeConfig(dbPath: string, filesDir: string, logFile?: string) {
  const env = { ...process.env }; if (!logFile) env.LCM_LOG_FILE_ENABLED = "false";
  return resolveLcmConfigWithDiagnostics(env, { ...ARM_PC, databasePath: dbPath, largeFilesDir: filesDir });
}
const logLines: Array<{ t_ms: number; turn: number | null; level: string; msg: string; fed_rows?: number }> = [];
function makeDeps(config: LcmConfig, over: Partial<LcmDependencies> = {}): LcmDependencies {
  const sink = (level: string) => (msg: string) => { logLines.push({ t_ms: now(), turn: clock.turn, level, msg }); };
  const log = createLcmLogger({ logger: { info: sink("host-info"), warn: sink("warn"), error: sink("error"), debug: sink("host-debug") },
    runtime: { logging: { shouldLogVerbose: () => true } } } as any, config);
  const tee = Object.fromEntries(Object.entries(log).map(([k, f]) => [k, (m: string) => { logLines.push({ t_ms: now(), turn: clock.turn, level: k, msg: m, fed_rows: fedRows }); (f as any)(m); }]));
  return {
    config, complete: zaiComplete, callGateway: async () => { throw new Error("gateway unavailable in the replay engine"); },
    // Host default model (OpenClaw would read its configured default); the summariser route is the z.ai adapter.
    resolveModel: () => ({ provider: "zai", model: ZAI_MODEL }),
    parseAgentSessionKey, isSubagentSessionKey: (k) => !!parseAgentSessionKey(k)?.suffix.startsWith("subagent:"),
    normalizeAgentId, buildSubagentSystemPrompt, readLatestAssistantReply, resolveAgentDir: () => process.env.HOME!,
    readVisibleSessionTranscriptMessageEntries: undefined, agentLaneSubagent: "subagent", log: tee as any, ...over,
  };
}

// ---- DB instrumentation (separate read-only connection; WAL readers never block the engine) ----
type Snap = { nodes: Record<string, any>; batches: Record<string, any>; summaries: Record<string, any>; ctx: Record<string, number> };
function snapshot(ro: DatabaseSync): Snap {
  const all = (q: string) => ro.prepare(q).all() as any[];
  const by = (rows: any[], k: string) => Object.fromEntries(rows.map((r) => [r[k], r]));
  try {
    return {
      nodes: by(all("SELECT node_id,batch_id,kind,depth,status,token_count,retry_count,failure_summary,created_at,ready_at,promoted_at FROM pending_summary_nodes"), "node_id"),
      batches: by(all("SELECT batch_id,status,created_at,published_at,failure_summary FROM pending_compaction_batches"), "batch_id"),
      summaries: by(all("SELECT summary_id,kind,depth,token_count,source_message_token_count,created_at FROM summaries"), "summary_id"),
      ctx: Object.fromEntries(all("SELECT item_type,count(*) n FROM context_items GROUP BY item_type").map((r) => [r.item_type, r.n])),
    };
  } catch { return { nodes: {}, batches: {}, summaries: {}, ctx: {} }; } // before migrations create the tables
}
const dbEvents: any[] = [];
let fedRows = 0; // rows (incl. the system row) fed to the engine so far: the kit-token axis position of each event
function diff(prev: Snap, cur: Snap, tokens: number | null) {
  const base = { t_ms: now(), turn: clock.turn, fed_rows: fedRows, tokens_at_observation: tokens };
  for (const [id, n] of Object.entries(cur.nodes)) if (prev.nodes[id]?.status !== n.status)
    dbEvents.push({ ...base, type: "pending_node", node_id: id, batch_id: n.batch_id, kind: n.kind, depth: n.depth, from: prev.nodes[id]?.status ?? null, to: n.status, token_count: n.token_count });
  for (const [id, b] of Object.entries(cur.batches)) if (prev.batches[id]?.status !== b.status)
    dbEvents.push({ ...base, type: "pending_batch", batch_id: id, from: prev.batches[id]?.status ?? null, to: b.status, failure: b.failure_summary });
  for (const [id, s] of Object.entries(cur.summaries)) if (!prev.summaries[id])
    dbEvents.push({ ...base, type: "summary_created", summary_id: id, kind: s.kind, depth: s.depth, token_count: s.token_count, source_tokens: s.source_message_token_count });
  if ((prev.ctx.summary ?? 0) !== (cur.ctx.summary ?? 0) || (prev.ctx.message ?? 0) !== (cur.ctx.message ?? 0))
    dbEvents.push({ ...base, type: "context_items", from: prev.ctx, to: cur.ctx });
}

// ---- public tools + delegated child loop behind deps.callGateway (r4 N2) ----
type Acct = { top_calls: number; top_tokens: number; deleg_llm_calls: number; deleg_tool_calls: number; deleg_tokens: number;
  deleg_readback: string[]; deleg_guard_hit: boolean; tool_log: any[] };
const toolSchema = (t: any) => ({ type: "function", function: { name: t.name, description: t.description, parameters: JSON.parse(JSON.stringify(t.parameters ?? {})) } });
async function runTool(tool: any, id: string, rawArgs: string) {
  let args: any; try { args = JSON.parse(rawArgs || "{}"); } catch { return JSON.stringify({ error: "arguments were not valid JSON" }); }
  try { const r = await tool.execute(id, args); return textOf(r?.content) || JSON.stringify(r?.details ?? r); }
  catch (e) { return JSON.stringify({ error: e instanceof Error ? e.message : String(e) }); }
}
/** Native tool-calling loop on GLM; returns the final text. Guard: GUARD tool calls, then one forced answer. */
async function toolLoop(purpose: string, msgs: ChatMessage[], tools: any[], onCall: (name: string, text: string, ms: number, args: string) => void, acct?: Acct) {
  const schemas = tools.map(toolSchema); let calls = 0, guardHit = false;
  for (;;) {
    const r = await chat(purpose, msgs, { maxTokens: READER_MAX_TOKENS, tools: guardHit ? undefined : schemas });
    if (acct && purpose === "delegated-child") { acct.deleg_llm_calls++; acct.deleg_readback.push(r.record.model_readback ?? "?"); }
    if (!r.toolCalls.length || guardHit) return { text: r.text, guardHit };
    msgs.push({ role: "assistant", content: r.text || null, tool_calls: r.toolCalls.map((c) => ({ id: c.id, type: "function", function: { name: c.name, arguments: c.args } })) });
    for (const c of r.toolCalls) {
      const t0 = performance.now(), tool = tools.find((t) => t.name === c.name);
      const out = tool ? await runTool(tool, c.id, c.args) : JSON.stringify({ error: `unknown tool ${c.name}` });
      calls++; onCall(c.name, out, Math.round(performance.now() - t0), c.args);
      msgs.push({ role: "tool", tool_call_id: c.id, content: out });
    }
    if (calls >= GUARD) { guardHit = true; msgs.push({ role: "user", content: `Tool budget exhausted (${GUARD} calls). Answer now.` }); }
  }
}
function gateway(engine: LcmContextEngine, deps: () => LcmDependencies, acct: Acct): LcmDependencies["callGateway"] {
  const runs = new Map<string, any>(), sessions = new Map<string, any>();
  return async ({ method, params = {} }) => {
    if (method === "agent") {
      const sk = String(params.sessionKey), runId = randomUUID(), run: any = { messages: [{ role: "user", content: String(params.message) }] };
      const child = { sessionKey: sk, deps: deps(), lcm: engine };
      const tools = [createLcmDescribeTool(child), createLcmExpandTool(child), createLcmGrepTool(child)]; // the granted lcm_expand path
      const msgs: ChatMessage[] = [{ role: "system", content: String(params.extraSystemPrompt ?? "") }, { role: "user", content: String(params.message) }];
      run.promise = toolLoop("delegated-child", msgs, tools, (name, text, ms, args) => {
        acct.deleg_tool_calls++; acct.deleg_tokens += estimateTokens(text);
        acct.tool_log.push({ level: "delegated", tool: name, args, result_chars: text.length, ms });
      }, acct).then((r) => { acct.deleg_guard_hit ||= r.guardHit; run.messages.push({ role: "assistant", content: r.text }); run.status = "ok"; },
        (e) => { run.status = "error"; run.error = e instanceof Error ? e.message : String(e); });
      runs.set(runId, run); sessions.set(sk, run); return { runId };
    }
    if (method === "agent.wait") {
      const run = runs.get(String(params.runId)); if (!run) return { status: "error", error: "unknown runId" };
      const t = Number(params.timeoutMs ?? 120_000);
      const won = await Promise.race([run.promise.then(() => "done"), sleep(t).then(() => "timeout")]);
      return won === "timeout" ? { status: "timeout" } : run.status === "ok" ? { status: "ok" } : { status: "error", error: run.error };
    }
    if (method === "sessions.get") return { messages: sessions.get(String(params.key))?.messages ?? [] };
    if (method === "sessions.delete") { sessions.delete(String(params.key)); return {}; }
    throw new Error(`Unsupported gateway method in LCM plugin: ${method}`);
  };
}

// ---- main ----
const P60 = PREFIX ? rd(join(MAT, "prefix60k.json")) : null;
const rows = jl(join(MAT, "transcript.jsonl")).filter((r, i) => (!SLICE || r.turn <= SLICE) && (!P60 || i <= P60.freeze_row_index));
const manifest = rd(join(MAT, "material.manifest.json")), facts = rd(join(MAT, "facts.json")), traps = rd(join(MAT, "traps.json"));
const RUN_NO = /^\d+$/.test(RUN) ? Number(RUN) : RUN, LAST = rows.length - 1;
const CPS = (arg("--checkpoints", "") || String(LAST)).split(",").map(Number).filter((r) => r <= LAST);
const sessionId = `s1-${SEED}-r${RUN}`, sessionKey = `agent:main:${sessionId}`;
const dbPath = join(RUN_DIR, "lcm.db"), filesDir = join(RUN_DIR, "lcm-files"), sessionFile = join(RUN_DIR, "session.jsonl");
const { config, diagnostics } = makeConfig(dbPath, filesDir, process.env.LCM_LOG_FILE);
const stray = Object.keys(process.env).filter((k) => k.startsWith("LCM_") && k !== "LCM_LOG_FILE");
if (stray.length) throw new Error(`default config required; stray LCM_* env: ${stray.join(",")}`);
const cfgView = { arm: ARM_LABEL, plugin_config_overrides: ARM_PC, freshTailMaxTokens: config.freshTailMaxTokens ?? "(unset)", contextThreshold: config.contextThreshold, threshold_tokens: Math.floor(config.contextThreshold * TOKEN_BUDGET),
  tokenBudget: TOKEN_BUDGET, freshTailCount: config.freshTailCount, leafChunkTokens: config.leafChunkTokens, leafTargetTokens: config.leafTargetTokens,
  leafMinFanout: config.leafMinFanout, condensedTargetTokens: config.condensedTargetTokens, condensedMinFanout: config.condensedMinFanout,
  summaryTimeoutMs: config.summaryTimeoutMs, summaryMaxOverageFactor: config.summaryMaxOverageFactor,
  proactiveThresholdCompactionMode: config.proactiveThresholdCompactionMode, largeFileTokenThreshold: config.largeFileTokenThreshold,
  delegationTimeoutMs: config.delegationTimeoutMs, maxExpandTokens: config.maxExpandTokens, summaryModel: config.summaryModel || "(host default: zai/glm-5.3)",
  expansionModel: config.expansionModel || "(unset: child runs on the host default, glm-5.3)" };
const materialView = { seed: SEED, slice_turns: SLICE || "all", rows: rows.length, roles: rows.reduce((a: any, r) => ((a[r.role] = (a[r.role] ?? 0) + 1), a), {}),
  smoke_suffix: manifest.smoke_suffix ?? null, cadence_s: CADENCE_S, checkpoints: CPS };
if (DRY) {
  const out = { pin: "988dee85592b9066ffe1c859542e8e18c23f2345", key_defaults: cfgView, material: materialView, sessionKey,
    config_sources: diagnostics, effective_config: { ...config, databasePath: dbPath, largeFilesDir: filesDir } };
  mkdirSync(join(TS, "s1", "dry-run"), { recursive: true });
  writeFileSync(join(TS, "s1", "dry-run", `${SEED}.config.json`), JSON.stringify(out, null, 2));
  console.log(JSON.stringify(out, null, 2)); process.exit(0);
}
// Logical store digest (row content per table); a file sha changes when a VACUUM INTO copy is reopened in WAL mode.
function logicalDigest(p: string) {
  const r = new DatabaseSync(p, { readOnly: true }), h = createHash("sha256");
  for (const { name } of r.prepare("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' AND name NOT LIKE '%fts%' ORDER BY name").all() as any[])
    h.update(name + JSON.stringify(r.prepare(`SELECT * FROM "${name}" ORDER BY rowid`).all()));
  r.close(); return h.digest("hex");
}
if (argv.includes("--wiring-check")) {
  // WIRING CHECK (not a probe answer): call the REAL lcm_expand_query on a clone of this run's checkpoint store so the
  // delegated child loop behind deps.callGateway runs once; plus the logical isolation digests of the probe clones.
  const main = join(RUN_DIR, "lcm.db"), dir = join(RUN_DIR, "clones", "wiring-expand"); mkdirSync(dir, { recursive: true });
  const src = new DatabaseSync(main, { readOnly: true }); const p = join(dir, "lcm.db"); src.exec(`VACUUM INTO '${p.replace(/'/g, "''")}'`); src.close();
  const acct: Acct = { top_calls: 0, top_tokens: 0, deleg_llm_calls: 0, deleg_tool_calls: 0, deleg_tokens: 0, deleg_readback: [], deleg_guard_hit: false, tool_log: [] };
  const { config: c } = makeConfig(p, join(RUN_DIR, "lcm-files")); const cdb = createLcmDatabaseConnection(p);
  let deps!: LcmDependencies; const eng = new LcmContextEngine((deps = makeDeps(c, { complete: async () => ({ content: [], error: { kind: "frozen" } }) })), cdb);
  deps.callGateway = gateway(eng, () => deps, acct);
  const tool = createLcmExpandQueryTool({ deps, lcm: eng, sessionId, sessionKey, requesterSessionKey: sessionKey });
  const params = { query: arg("--query", "fixture"), prompt: arg("--prompt", "What is the current request and the active constraint stated at session start?") };
  const t0 = performance.now(); clock.t0 = Date.now();
  const out = await tool.execute("wiring-1", params);
  const res = { label: "WIRING CHECK - lcm_expand_query called directly by the harness, not by a reader; not a probe answer", params,
    wall_s: Math.round(performance.now() - t0) / 1000, result: (out as any)?.details ?? textOf((out as any)?.content), accounting: acct,
    child_calls: callLog.map((x) => ({ purpose: x.purpose, latency_ms: x.latency_ms, model: x.model_readback, finish: x.finish_reason, error: x.error })),
    isolation_logical: { main: logicalDigest(main), clones: Object.fromEntries(readdirSync(join(RUN_DIR, "clones")).filter((d) => d !== "wiring-expand")
      .map((d) => [d, logicalDigest(join(RUN_DIR, "clones", d, "lcm.db"))])) } };
  closeLcmConnection(cdb);
  writeFileSync(join(RUN_DIR, "wiring-expand-check.json"), JSON.stringify(res, null, 2));
  console.log(JSON.stringify(res, null, 2)); process.exit(0);
}
if (argv.includes("--grep-check")) {
  // GREP CHECK (round 2): re-run the reader's RECORDED lcm_grep calls by hand through the real tool on fresh clones of the
  // checkpoint, under the fixed payload dir and under round 1's copied-dir clone config. CONTROL rows are labelled.
  const main = join(RUN_DIR, "lcm.db"), id = arg("--probe", "S1-F11-2")!, fact = facts.find((f: any) => f.id === id);
  const recorded = rd(join(RUN_DIR, "clones", `open-${id}`, "answer.json")).accounting.tool_log.filter((l: any) => l.level === "top" && l.tool === "lcm_grep");
  const hit = (t: string) => { const body = t.replace(/^\*\*Pattern:\*\*.*$/m, ""); return { value: body.includes(fact.value), tag: body.includes(`[${id}]`) }; }; // pattern echo excluded
  const out: any = { label: "GREP CHECK - recorded reader calls re-run by hand; CONTROL rows are diagnostics, not probe answers", probe: id, value: fact.value, variants: [] };
  const variants: Array<[string, string]> = [["payload dir = run lcm-files (fixed clone config)", filesDir], ["payload dir = copied dir (round-1 clone config)", join(RUN_DIR, "clones", `open-${id}`, "lcm-files")]];
  for (const [i, [name, fdir]] of variants.entries()) {
    const dir = join(RUN_DIR, "clones", `grep-check-${i}`); mkdirSync(dir, { recursive: true });
    const src = new DatabaseSync(main, { readOnly: true }), p = join(dir, "lcm.db"); src.exec(`VACUUM INTO '${p.replace(/'/g, "''")}'`); src.close();
    const cdb = createLcmDatabaseConnection(p), eng = new LcmContextEngine(makeDeps(makeConfig(p, fdir).config, { complete: async () => ({ content: [], error: { kind: "frozen" } }) }), cdb);
    const ctx = { deps: makeDeps(makeConfig(p, fdir).config), lcm: eng, sessionId, sessionKey }, grep = createLcmGrepTool(ctx), describe = createLcmDescribeTool(ctx);
    const fileId = (cdb.prepare("SELECT file_id FROM large_files").all() as any[]).map((r) => r.file_id)[0];
    const v: any = { name, largeFilesDir: fdir, recorded: [], control: [] };
    for (const l of recorded) { const t = await runTool(grep, "gc", l.args); v.recorded.push({ args: JSON.parse(l.args), result_chars: t.length, round1_result_chars: l.result_chars, ...hit(t), result: t.slice(0, 600) }); }
    for (const [tool, a] of [[grep, { pattern: id, mode: "regex", scope: "files" }], [grep, { pattern: fact.value, mode: "full_text", scope: "files" }], [describe, { id: fileId }], [describe, { id: fileId, expandFile: true }]] as const) {
      const t = await runTool(tool, "gc", JSON.stringify(a)); v.control.push({ label: "CONTROL", tool: tool.name, args: a, result_chars: t.length, ...hit(t), result: t.slice(0, 400) });
    }
    if (i === 0) {
      const asm = await eng.assemble({ sessionId, sessionKey, messages: rows.filter((r) => r.role !== "system").map(toAgentMessage), tokenBudget: TOKEN_BUDGET });
      const ctxText = asm.messages.map((m: any) => textOf(m.content)).join("\n");
      v.assembled = { messages: asm.messages.length, file_id: fileId, file_id_in_context: ctxText.includes(fileId), fact_in_context: ctxText.includes(fact.value),
        stub_excerpt: ctxText.slice(Math.max(0, ctxText.indexOf(fileId) - 40), ctxText.indexOf(fileId) + 120) };
    }
    closeLcmConnection(cdb); v.clone_logical_equals_main = logicalDigest(p) === logicalDigest(main); out.variants.push(v);
  }
  writeFileSync(join(RUN_DIR, "grep-check.json"), JSON.stringify(out, null, 2));
  console.log(JSON.stringify(out, (k, x) => (k === "result" ? undefined : x), 2)); process.exit(0);
}
if (existsSync(RUN_DIR) && readdirSync(RUN_DIR).some((f) => f !== "home" && f !== "lcm.log")) throw new Error(`run dir not empty: ${RUN_DIR}`);
mkdirSync(RUN_DIR, { recursive: true }); writeFileSync(sessionFile, "");
writeFileSync(join(RUN_DIR, "config.json"), JSON.stringify({ key_defaults: cfgView, effective_config: config }, null, 2));
const db = createLcmDatabaseConnection(dbPath);
const engine = new LcmContextEngine(makeDeps(config), db);
const systemSlot = rows.filter((r) => r.role === "system").map((r) => r.content).join("\n");
const cont = manifest.continuity as any[];
const contCheck = (msgs: any[], sys: string) => Object.fromEntries(cont.map((c) => [c.id,
  c.row_role === "system" ? sys.includes(c.value) : msgs.some((m) => textOf(m.content).includes(c.value))]));
const turns: any[] = [], visible: any[] = [], rowIndexOf = new Map<any, number>();
const byTurn = [...new Set(rows.map((r) => r.turn))];
let ro: DatabaseSync | null = null, prev: Snap = { nodes: {}, batches: {}, summaries: {}, ctx: {} }, lastTokens: number | null = null;
const poll = () => { if (!ro && existsSync(dbPath)) ro = new DatabaseSync(dbPath, { readOnly: true }); if (ro) { const c = snapshot(ro); diff(prev, c, lastTokens); prev = c; } };
const poller = setInterval(poll, 500);
// S6 A2: after every publication, the assembled messages the next model call receives -> continuity/event-<k>.json
const savedPub = new Set<string>(); let contIdx = 0, p60: any = null;
// A compaction with no prepared batch runs the engine's legacy fallback ("compact: done ... compacted=true", no
// pending batch): recorded from the engine's own log line (id = createdSummaryId), deduplicated, as a direct event.
const directDone = () => [...new Map(logLines.filter((l) => /\[lcm\] compact: done .*compacted=true/.test(l.msg) && !dbEvents.some((e) =>
  e.type === "pending_batch" && e.to === "published" && Math.abs(e.t_ms - l.t_ms) < 5000)).map((l) => {
  const g = (k: string) => l.msg.match(new RegExp(`${k}=(\\S+)`))?.[1];
  return [g("createdSummaryId") ?? `compact@${l.t_ms}`, { id: g("createdSummaryId") ?? `compact@${l.t_ms}`, t_ms: l.t_ms, turn: l.turn, fed_rows: l.fed_rows,
    duration_ms: Number(String(g("duration")).replace("ms", "")), tokens_before: Number(g("tokensBefore")), tokens_after: Number(g("tokensAfter")) }] as const;
})).values()];
const publishedIds = (cut = Infinity) => [...dbEvents.filter((e) => e.type === "pending_batch" && e.to === "published" && e.t_ms <= cut).map((e) => e.batch_id as string),
  ...directDone().filter((d) => d.t_ms <= cut).map((d) => d.id)];
const CONT_SRC = "assemble() output (system slot + messages) = what the next model call receives";
const asmText = (asm: any) => [asm.systemPromptAddition, systemSlot].filter(Boolean).join("\n\n") + "\n" + asm.messages.map((m: any) => textOf(m.content)).join("\n");
function saveContinuity(asm: any, pubs: string[], label: string) {
  const fresh = pubs.filter((b) => !savedPub.has(b)); if (!fresh.length) return;
  fresh.forEach((b) => savedPub.add(b)); mkdirSync(join(RUN_DIR, "continuity"), { recursive: true });
  writeFileSync(join(RUN_DIR, "continuity", `event-${contIdx}.json`), JSON.stringify({ event: contIdx++, batch_ids: fresh, label, turn: clock.turn,
    t_ms: now(), source: CONT_SRC, text: asmText(asm) }));
}
// S7 D1: a consistent snapshot of the live store (VACUUM INTO on the read-only connection) + the host view at that row
const cpSnaps: any[] = [];
function takeCp(row: number) {
  poll(); const dir = join(RUN_DIR, `cp-${row}`), p = join(dir, "lcm.db"); mkdirSync(dir, { recursive: true });
  ro!.exec(`VACUUM INTO '${p.replace(/'/g, "''")}'`);
  cpSnaps.push({ row, dir, db: p, visible: visible.slice(), t_ms: now(), turn: clock.turn, turns_n: turns.length, fed_rows: fedRows });
}
clock.t0 = Date.now();
for (const [i, turn] of byTurn.entries()) {
  // S7: a catch-up turn after an overrun starts >= 1.1 s after the previous one (a host turn is never sub-second); without it,
  // back-to-back turns put >= 3 repeated user rows in one createdAt second and the store's replay-flood guard refuses them.
  const scheduled = i * CADENCE_S * 1000; await sleep(Math.max(scheduled, (turns.at(-1)?.start_ms ?? -2000) + 1100) - now());
  const start = now(); clock.turn = turn;
  const turnRows = rows.filter((r) => r.turn === turn && r.role !== "system");
  const t0 = performance.now();
  for (const r of turnRows) { const m = toAgentMessage(r); await engine.ingest({ sessionId, sessionKey, message: m }); visible.push(m); rowIndexOf.set(m, rows.indexOf(r)); fedRows = rows.indexOf(r) + 1; }
  const ingestMs = Math.round(performance.now() - t0), a0 = performance.now();
  const asm = await engine.assemble({ sessionId, sessionKey, messages: visible, tokenBudget: TOKEN_BUDGET });
  const assembleMs = Math.round(performance.now() - a0);
  poll(); // events done by now (incl. debt the assemble drained pre-assembly) are in this assemble's output
  saveContinuity(asm, publishedIds(), "this turn's assemble (first model call after the event)");
  lastTokens = asm.estimatedTokens + estimateTokens(systemSlot);
  const f0 = performance.now();
  const budget = PREFIX && i === byTurn.length - 1 ? Math.floor((lastTokens - 1) / config.contextThreshold) : TOKEN_BUDGET;
  if (budget !== TOKEN_BUDGET) p60 = { ...P60, turn, fed_rows: fedRows, tokens_at_trigger: lastTokens, lowered_token_budget: budget,
    threshold_tokens: Math.floor(config.contextThreshold * budget), trigger_t_ms: now(), path: "afterTurn -> evaluatePostTurnCompaction (deferred default path)" };
  await engine.afterTurn({ sessionId, sessionKey, sessionFile, messages: visible, prePromptMessageCount: visible.length, tokenBudget: budget, currentTokenCount: lastTokens });
  const afterTurnMs = Math.round(performance.now() - f0); poll();
  turns.push({ turn, scheduled_ms: scheduled, start_ms: start, lag_ms: start - scheduled, rows: turnRows.length, last_row: rows.indexOf(turnRows.at(-1)),
    ingest_ms: ingestMs, assemble_ms: assembleMs, after_turn_ms: afterTurnMs, assembled_tokens: lastTokens, assembled_messages: asm.messages.length,
    continuity: contCheck(asm.messages, systemSlot), pending_nodes: Object.values(prev.nodes).map((n: any) => n.status),
    host_visible_wait: "n/a (engine-direct: no host model turn; assemble_ms + after_turn_ms are the engine's blocking share)" });
  for (const r of CPS) if (r < LAST && r < fedRows && !cpSnaps.some((c) => c.row === r)) takeCp(r);
}
// Smoke suffix (r4 F1): forced compaction after the manifest's row, WIRING-ONLY. Natural events are recorded above.
let forced: any = null;
const suffix = manifest.smoke_suffix;
if (suffix && rows.some((r) => r.id === suffix.row_id)) {
  const s = now(), t = performance.now();
  let result: any; try { result = await engine.compact({ sessionId, sessionKey, sessionFile, tokenBudget: TOKEN_BUDGET, currentTokenCount: lastTokens ?? undefined, force: true }); }
  catch (e) { result = { ok: false, error: e instanceof Error ? e.message : String(e) }; }
  poll();
  const asm = await engine.assemble({ sessionId, sessionKey, messages: visible, tokenBudget: TOKEN_BUDGET });
  forced = { label: "FORCED (smoke suffix S1-SMOKE-EVENT) - WIRING-ONLY", after_row: suffix.row_index, fed_rows: fedRows, start_ms: s, wall_ms: Math.round(performance.now() - t),
    tokens_at_trigger: lastTokens, result: JSON.parse(JSON.stringify(result ?? null)), assembled_tokens_after: asm.estimatedTokens + estimateTokens(systemSlot),
    continuity_after: contCheck(asm.messages, systemSlot) };
}
if (p60) { // the drain runs in the background: wait until a publication landed and no node/batch is in flight (cap 15 min)
  const busy = () => Object.values(prev.nodes).some((n: any) => ["planned", "running"].includes(n.status)) ||
    Object.values(prev.batches).some((b: any) => ["planning", "publishing"].includes(b.status));
  const n0 = publishedIds().length;
  for (let quiet = 0; quiet < 4 && now() - p60.trigger_t_ms < 900_000;) { await sleep(500); poll(); quiet = publishedIds().length > n0 && !busy() ? quiet + 1 : 0; }
  p60.waited_ms = now() - p60.trigger_t_ms; p60.events_after_trigger = publishedIds().length - n0;
}
clearInterval(poller); if (CPS.includes(LAST)) takeCp(LAST); clock.turn = null;
const factById = new Map<string, any>(facts.map((f: any) => [f.id, f])), trapById = new Map<string, any>(traps.map((t: any) => [t.id, t]));
const parseObj = (t: string) => { const s = (t || "").replace(/^```(?:json)?|```$/gm, "").trim(); const i = s.indexOf("{");
  try { return i >= 0 ? JSON.parse(s.slice(i, s.lastIndexOf("}") + 1)) : null; } catch { return null; } };
const recallPath = (a: Acct) => a.tool_log.some((l) => l.tool === "lcm_expand_query") ? (a.deleg_llm_calls ? "delegated" : "expand_query called, delegation did not run") : "grep/describe only (DIAGNOSTIC)";
const policy = OPEN || OPEN_PROBES.length ? recallPolicyPrompt() : "";
const usageOf = (cs: typeof callLog) => cs.map((c: any) => ({ purpose: c.purpose, prompt_tokens: c.usage?.prompt_tokens ?? null,
  completion_tokens: c.usage?.completion_tokens ?? null, latency_ms: c.latency_ms, model: c.model_readback, error: c.error }));
async function finishCp(cp: any) {
const fcd = cp.row === LAST ? forced : null; // the smoke suffix belongs to the last checkpoint only
const cdb0 = new DatabaseSync(cp.db, { readOnly: true }), checkpointMs = now();
// ---- receipts, admission, events (all recorded BEFORE any probe) ----
const q = (sql: string, ...p: any[]) => cdb0.prepare(sql).all(...p) as any[];
const stored = q("SELECT message_id,role,content FROM messages ORDER BY seq");
const files = q("SELECT file_id,storage_uri,byte_size FROM large_files").map((f) => ({ ...f, text: existsSync(f.storage_uri) ? readFileSync(f.storage_uri, "utf8") : "" }));
const inStore = (v: string) => stored.some((m) => m.content.includes(v)) || files.some((f) => f.text.includes(v));
const sliceFacts = facts.filter((f: any) => f.row_index < cp.fed_rows);
const admission = { rows_fed: materialView.roles, stored_by_role: q("SELECT role,count(*) n FROM messages GROUP BY role"),
  facts_planted_in_slice: sliceFacts.length, facts_admitted: sliceFacts.filter((f: any) => inStore(f.value)).length,
  facts_missing: sliceFacts.filter((f: any) => !inStore(f.value)).map((f: any) => f.id), large_files: files.map((f) => ({ file_id: f.file_id, bytes: f.byte_size })) };
const summaries = q("SELECT summary_id,kind,depth,token_count,content,created_at FROM summaries");
// S7 D3: a summary's publication = its prepared batch (pending node -> canonical_summary_id), else the direct (legacy
// fallback) compaction whose "compact: done" line is the first at or after the summary's created_at (1 s resolution).
const cut = dbEvents.filter((e) => e.t_ms <= cp.t_ms), directs = directDone().filter((d) => d.t_ms <= cp.t_ms).sort((a, b) => a.t_ms - b.t_ms);
const batchOf = new Map(q("SELECT canonical_summary_id s, batch_id b FROM pending_summary_nodes WHERE canonical_summary_id IS NOT NULL").map((r) => [r.s, r.b]));
const pubOf = (s: any) => batchOf.get(s.summary_id) ?? directs.find((d) => clock.t0 + d.t_ms >= Date.parse(s.created_at.replace(" ", "T") + "Z"))?.id ?? `created@${s.created_at}`;
const receipts = (manifest.receipt_targets as any[]).map((t) => {
  if (t.kind === "externalization") {
    const f = files.find((x) => x.text.includes(`[${t.id}]`));
    return { ...t, status: f ? "PRESENT" : "UNTESTED (not externalized)", file_id: f?.file_id ?? null, bytes: f?.byte_size ?? null };
  }
  const msgIds = stored.filter((m) => m.content.includes(`[${t.id}]`)).map((m) => m.message_id);
  const leaves = msgIds.length ? q(`SELECT DISTINCT summary_id FROM summary_messages WHERE message_id IN (${msgIds.join(",")})`).map((r) => r.summary_id) : [];
  const chain = new Set<string>(leaves);
  for (let grew = true; grew;) { grew = false; for (const r of q("SELECT summary_id,parent_summary_id FROM summary_parents"))
    if (chain.has(r.parent_summary_id) && !chain.has(r.summary_id)) { chain.add(r.summary_id); grew = true; } }
  const pubs = [...new Set(summaries.filter((s) => chain.has(s.summary_id)).map(pubOf))];
  return { ...t, status: pubs.length >= t.required_events ? "PASS" : "UNTESTED", chain: [...chain], publications: pubs, publication_events_seen: pubs.length,
    rule: "publications = distinct compactions (prepared batch or direct) that created a summary in the chain (store summaries + summary_parents)" };
});
// S7 D5: preparation wall from the store: batch created_at (1 s resolution) -> the batch's last node ready_at (ms)
const storeTiming = Object.fromEntries(q("SELECT b.batch_id,b.created_at,b.published_at,MAX(n.ready_at) ready_at FROM pending_compaction_batches b LEFT JOIN pending_summary_nodes n ON n.batch_id=b.batch_id GROUP BY b.batch_id")
  .map((r) => [r.batch_id, { ...r, wall_s: r.ready_at ? (Date.parse(r.ready_at) - Date.parse(r.created_at.replace(" ", "T") + "Z")) / 1000 : null,
    source: "store: pending_compaction_batches.created_at (1 s) -> MAX(pending_summary_nodes.ready_at) (ms)" }]));
// ---- probes, each from a CLONE of the checkpoint snapshot (VACUUM INTO); clone engines are frozen ----
const dbShaBefore = sha(cp.db), results: any[] = [], probeMeta: any[] = [], readerCalls: any[] = [];
const frozenComplete: LcmDependencies["complete"] = async () => { probeMeta.push({ frozen_complete_attempt: now() }); return { content: [], error: { kind: "frozen", message: "clone engines never summarise" } }; };
async function openClone(name: string) {
  const dir = join(cp.dir, "clones", name); mkdirSync(dir, { recursive: true });
  const p = join(dir, "lcm.db"); cdb0.exec(`VACUUM INTO '${p.replace(/'/g, "''")}'`);
  // Payloads stay in the run's own largeFilesDir (the dir every storage_uri names; write-once, read-only here): the
  // plugin refuses reads outside largeFilesDir (summary-store.ts:1887-1891). Round 1 copied the dir -> file reads null.
  const { config: c } = makeConfig(p, filesDir);
  const sha0 = logicalDigest(p), cdb = createLcmDatabaseConnection(p);
  const acct: Acct = { top_calls: 0, top_tokens: 0, deleg_llm_calls: 0, deleg_tool_calls: 0, deleg_tokens: 0, deleg_readback: [], deleg_guard_hit: false, tool_log: [] };
  let deps!: LcmDependencies; const eng = new LcmContextEngine((deps = makeDeps(c, { complete: frozenComplete })), cdb);
  deps.callGateway = gateway(eng, () => deps, acct);
  const asm = await eng.assemble({ sessionId, sessionKey, messages: cp.visible, tokenBudget: TOKEN_BUDGET });
  return { dir, p, cdb, eng, deps, acct, asm, sha0 };
}
// The checkpoint assemble (what the probes see), on a frozen clone: continuity files up to this checkpoint + its own record
const ca = await openClone("checkpoint-assemble"), cpAsm = ca.asm;
const cdir = join(cp.dir, "continuity"), rdir = join(RUN_DIR, "continuity"); mkdirSync(cdir, { recursive: true });
const prior = existsSync(rdir) ? readdirSync(rdir).map((f) => rd(join(rdir, f))).filter((e) => e.t_ms <= cp.t_ms) : [];
prior.forEach((e) => writeFileSync(join(cdir, `event-${e.event}.json`), JSON.stringify(e)));
const freshPubs = publishedIds(cp.t_ms).filter((b) => !prior.some((e) => e.batch_ids.includes(b)));
if (freshPubs.length) writeFileSync(join(cdir, `event-${prior.length}.json`), JSON.stringify({ event: prior.length, batch_ids: freshPubs,
  label: "checkpoint assemble (what the probes see)", turn: cp.turn, t_ms: cp.t_ms, source: CONT_SRC, text: asmText(cpAsm) }));
const cpContinuity = { t_ms: cp.t_ms, continuity: contCheck(cpAsm.messages, systemSlot) }; // strict check at the checkpoint
writeFileSync(join(cp.dir, "reader-view.txt"), asmText(cpAsm)); // S8: what the probes see, always (per-fact loss class)
await ca.eng.dispose?.(); closeLcmConnection(ca.cdb);
const eventsOut = { population: POP, checkpoint: { row: cp.row, turn: cp.turn, t_ms: cp.t_ms, fed_rows: cp.fed_rows }, prefix60k: p60, direct_compactions: directs,
  checkpoint_continuity: cpContinuity, store_timing: storeTiming, natural_publications: cut.filter((e) => e.type === "pending_batch" && e.to === "published" && e.t_ms < (fcd?.start_ms ?? Infinity)).length,
  natural_preparations: new Set(cut.filter((e) => e.type === "pending_node" && e.t_ms < (fcd?.start_ms ?? Infinity)).map((e) => e.batch_id)).size,
  forced: fcd, db_events: cut, summariser_calls: callLog.filter((c) => c.purpose === "summariser" && c.t_start_ms <= cp.t_ms).map((c) => ({ ...c, population: POP })),
  levels: summaries.map((s) => ({ summary_id: s.summary_id, kind: s.kind, depth: s.depth, token_count: s.token_count, publication: pubOf(s),
    level: s.content.includes(FALLBACK_SUMMARY_MARKER) ? "fallback" : "normal|aggressive (not distinguished at the store)" })),
  escalation_log_lines: logLines.filter((l) => l.t_ms <= cp.t_ms && /aggressive|fallback|capped|timeout|circuit/i.test(l.msg)).map((l) => ({ t_ms: l.t_ms, level: l.level, msg: l.msg.slice(0, 300) })) };
writeFileSync(join(cp.dir, "events.json"), JSON.stringify(eventsOut, null, 2));
writeFileSync(join(cp.dir, "turns.json"), JSON.stringify(turns.slice(0, cp.turns_n), null, 2));
const receiptById = new Map(receipts.map((r: any) => [r.id, r]));
function row(p: any, answers: any, meta: any, extra: any) {
  const f = factById.get(p.id), tr = trapById.get(p.id), a = answers?.[p.id];
  return { schema: "s1-result-v1", kind: "probe", probe_id: p.id, probe_kind: p.kind, expect: p.expect, gold: f?.value ?? tr?.answer ?? p.gold ?? null,
    stale: f?.stale ?? null, fact_class: f?.class ?? null, placement: f?.placement ?? null, row_role: f?.row_role ?? null,
    answer: typeof a === "string" ? a : a == null ? null : JSON.stringify(a), answered: a != null, timed_out: false, error: meta.error ?? null,
    status: meta.error ? "ERROR" : "OK", arm: ARM_LABEL, seed: SEED, run: RUN_NO, run_id: `${ARM_LABEL}/${SEED}/r${RUN}/${clock.t0}`, checkpoint_row: cp.row,
    lane: "glm", reader: "glm", reader_readback: { model: meta.model_readback ?? null, effort: "n/a (glm-5.3 lane sends no effort input)" },
    store_backed: true, receipt: receiptById.get(p.id) ? { kind: receiptById.get(p.id).kind, status: receiptById.get(p.id).status } : null,
    context_format: meta.context_format, summary_blocks: meta.summary_blocks, timing_label: "WIRING-ONLY", ...extra };
}
async function ask(purpose: string, system: string, asm: any, user: string, tools?: any[], acct?: Acct) {
  const native: ChatMessage[] = [{ role: "system", content: system }, ...toChat(asm.messages), { role: "user", content: user }];
  const summary_blocks = toChat(asm.messages).reduce((n, m) => n + ((m.content ?? "").match(/<summary[\s>]/g)?.length ?? 0), 0); // D7: in the request built here
  const t0 = performance.now(); let fmt = "native-roles", text = "", guard = false, error: string | undefined;
  const onCall = (name: string, out: string, ms: number, args: string) => { acct!.top_calls++; acct!.top_tokens += estimateTokens(out); acct!.tool_log.push({ level: "top", tool: name, args, result_chars: out.length, ms }); };
  for (const msgs of [native, [{ role: "system", content: system }, { role: "user", content: flatten(toChat(asm.messages)) + "\n\n" + user }]]) {
    try {
      if (tools) ({ text, guardHit: guard } = await toolLoop(purpose, msgs, tools, onCall, acct));
      else text = (await chat(purpose, msgs, { maxTokens: READER_MAX_TOKENS })).text;
      error = undefined; break;
    } catch (e) { error = e instanceof Error ? e.message : String(e); if (!/HTTP 400/.test(error)) break; fmt = "flattened (native rejected: HTTP 400)"; }
  }
  const last = [...callLog].reverse().find((c) => c.purpose === purpose);
  return { text, error, guard, context_format: fmt, summary_blocks, wall_s: Math.round(performance.now() - t0) / 1000, model_readback: last?.model_readback };
}
// S6 A1: + the continuation batch <prefix>-BCONT (one probe per continuation.json field), shaped as S2 builds it
const cont0 = rd(join(MAT, "continuation.json")), pb = jl(join(MAT, "probe_batches.jsonl"));
const batches = PREFIX ? [] : [...pb, { id: pb[0].id.replace(/-[^-]+$/, "") + "-BCONT", text: pb[0].text, probes: Object.keys(cont0)
  .filter((k) => !["id", "row_id", "row_index", "row_role"].includes(k)).map((k) => ({ id: `${cont0.id}.${k}`, kind: "continuation_field",
    expect: "value", gold: cont0[k], text: `For the pending mid-task continuation, what is its \`${k}\`?` })) }].slice(0, BATCHES);
for (const b of batches) {
  const c = await openClone(b.id), k0 = callLog.length;
  const ctx = { deps: c.deps, lcm: c.eng, sessionId, sessionKey }; // D6 --open: the public tools on this clone, every batch
  const tools = OPEN ? [createLcmGrepTool(ctx), createLcmDescribeTool(ctx), createLcmExpandQueryTool({ ...ctx, requesterSessionKey: sessionKey })] : undefined;
  const sys = [policy && OPEN ? policy : "", c.asm.systemPromptAddition, systemSlot].filter(Boolean).join("\n\n");
  const user = b.text + "\n" + b.probes.map((p: any) => `${p.id}: ${p.text}`).join("\n");
  const r = await ask(OPEN ? "reader-open" : "reader", sys, c.asm, user, tools, c.acct);
  const answers = parseObj(r.text), calls = usageOf(callLog.slice(k0).filter((x) => x.purpose !== "summariser")); readerCalls.push({ batch: b.id, calls });
  writeFileSync(join(c.dir, "answer.json"), JSON.stringify({ raw_reply: r.text, meta: { ...r, text: undefined }, reader_calls: calls, accounting: OPEN ? c.acct : undefined,
    assembled_messages: c.asm.messages.length, assembled_tokens: c.asm.estimatedTokens }, null, 2));
  const openExtra = OPEN ? { arm_kind: "open", attribution: "batch", tool_calls: c.acct.top_calls, delegated_llm_calls: c.acct.deleg_llm_calls, delegated_tool_calls: c.acct.deleg_tool_calls,
    tokens_read_back: c.acct.top_tokens, runaway_guard_hit: r.guard, recall_path: c.acct.top_calls ? recallPath(c.acct) : "context" } : {};
  for (const p of b.probes) results.push(row(p, answers, r, { arm_kind: "plain", batch_id: b.id, clone_id: `clones/${b.id}`, answer_wall_s_batch: r.wall_s, attribution: "batch", ...openExtra }));
  await c.eng.dispose?.(); closeLcmConnection(c.cdb); probeMeta.push({ clone: b.id, logical_unchanged: logicalDigest(c.p) === c.sha0 });
}
for (const id of PREFIX ? [] : OPEN_PROBES) {
  const p = batches.flatMap((b: any) => b.probes).find((x: any) => x.id === id) ?? jl(join(MAT, "probe_batches.jsonl")).flatMap((b: any) => b.probes).find((x: any) => x.id === id);
  if (!p) throw new Error(`unknown open probe ${id}`);
  const c = await openClone(`open-${id}`);
  const ctx = { deps: c.deps, lcm: c.eng, sessionId, sessionKey };
  const tools = [createLcmGrepTool(ctx), createLcmDescribeTool(ctx), createLcmExpandQueryTool({ ...ctx, requesterSessionKey: sessionKey })];
  const sys = [policy, c.asm.systemPromptAddition, systemSlot].filter(Boolean).join("\n\n");
  const r = await ask("reader-open", sys, c.asm, `Reply only as a JSON object mapping each probe id to its answer string (use I don't know for ABSTAIN).\n${p.id}: ${p.text}`, tools, c.acct);
  const answers = parseObj(r.text);
  writeFileSync(join(c.dir, "answer.json"), JSON.stringify({ raw_reply: r.text, meta: { ...r, text: undefined }, accounting: c.acct }, null, 2));
  results.push(row(p, answers, r, { arm: `${ARM}-open`, arm_kind: "open", batch_id: null, clone_id: `clones/open-${id}`, attribution: "answer",
    tool_calls: c.acct.top_calls, delegated_llm_calls: c.acct.deleg_llm_calls, delegated_tool_calls: c.acct.deleg_tool_calls,
    tokens_read_back: c.acct.top_tokens, delegated_tokens_read_back: c.acct.deleg_tokens, delegated_readback: [...new Set(c.acct.deleg_readback)],
    answer_wall_s: r.wall_s, runaway_guard_hit: r.guard, delegated_guard_hit: c.acct.deleg_guard_hit, recall_path: recallPath(c.acct) }));
  await c.eng.dispose?.(); closeLcmConnection(c.cdb); probeMeta.push({ clone: `open-${id}`, logical_unchanged: logicalDigest(c.p) === c.sha0, recall_policy_sha256: createHash("sha256").update(policy).digest("hex") });
}
writeFileSync(join(cp.dir, "results.jsonl"), results.map((r) => JSON.stringify(r)).join("\n") + (results.length ? "\n" : ""));
const dbShaAfter = sha(cp.db); cdb0.close();
const usage = (cs: typeof callLog) => ({ calls: cs.length, errors: cs.filter((c) => c.error).length, latency_ms: cs.map((c) => c.latency_ms),
  prompt_tokens: cs.reduce((n, c: any) => n + (c.usage?.prompt_tokens ?? 0), 0), completion_tokens: cs.reduce((n, c: any) => n + (c.usage?.completion_tokens ?? 0), 0) });
const probeCalls = readerCalls.length ? callLog.filter((c) => c.t_start_ms >= checkpointMs) : [];
const summary = { arm: ARM_LABEL, pin: "988dee8", seed: SEED, run: RUN_NO, population: POP, prefix60k: p60, sessionKey, config: cfgView, material: materialView,
  checkpoint: eventsOut.checkpoint, checkpoint_ms: checkpointMs, turns_cadence: { declared_s: CADENCE_S, max_lag_ms: Math.max(...turns.map((t) => t.lag_ms)) },
  events: { natural_publications: eventsOut.natural_publications, natural_preparations: eventsOut.natural_preparations, direct_compactions: directs.length, forced: fcd && { ...fcd, result: fcd.result } },
  admission, receipts, store_summaries: summaries.length, continuity_final: fcd?.continuity_after ?? turns[cp.turns_n - 1]?.continuity,
  summary_blocks_in_reader_request: [...new Set(results.map((r) => r.summary_blocks))],
  calls: { summariser: usage(eventsOut.summariser_calls as any), reader: usage(probeCalls.filter((c) => c.purpose === "reader")),
    reader_open: usage(probeCalls.filter((c) => c.purpose === "reader-open")), delegated_child: usage(probeCalls.filter((c) => c.purpose === "delegated-child")) },
  reader_calls: readerCalls, isolation: { snapshot_sha256_before_probes: dbShaBefore, snapshot_sha256_after_probes: dbShaAfter, clones: probeMeta },
  log: { lines: logLines.length, warn: logLines.filter((l) => l.level === "warn").length, error: logLines.filter((l) => l.level === "error").length } };
writeFileSync(join(cp.dir, "summary.json"), JSON.stringify(summary, null, 2));
console.log(JSON.stringify({ checkpoint: eventsOut.checkpoint, events: summary.events, summary_blocks: summary.summary_blocks_in_reader_request,
  receipts: receipts.map((r: any) => ({ id: r.id, kind: r.kind, status: r.status })) }, null, 2));
}
for (const cp of cpSnaps) await finishCp(cp);
await engine.dispose?.(); ro?.close(); closeLcmConnection(db);
writeFileSync(join(RUN_DIR, "host-log.jsonl"), logLines.map((l) => JSON.stringify(l)).join("\n") + "\n");
process.exit(0);
