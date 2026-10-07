// S1 z.ai adapter: the lossless-claw `CompleteFn` seam (src/types.ts:64-77) and the GLM reader, both on the
// GLM coding plan (subscription lane). GLM_API_KEY is read from the environment into process memory only; it is
// never printed, logged or written (the call log below records sizes, usage and timing, never headers or keys).
import type { CompleteFn } from "track-s-lossless/src/types.js";

export const ZAI_URL = "https://api.z.ai/api/coding/paas/v4/chat/completions";
export const ZAI_MODEL = "glm-5.3";
const HTTP_TIMEOUT_MS = 300_000; // transport guard only; the engine enforces its own summaryTimeoutMs (60 s)

let cachedKey: string | undefined;
function key(): string {
  if (cachedKey) return cachedKey;
  cachedKey = process.env.GLM_API_KEY;
  if (!cachedKey) throw new Error("GLM_API_KEY is required in the environment");
  return cachedKey;
}

export type CallRecord = {
  purpose: string; t_start_ms: number; latency_ms: number; model_readback: string | null;
  max_tokens: number | null; prompt_chars: number; reply_chars: number; usage: unknown;
  finish_reason: string | null; tool_calls: number; error: string | null; turn: number | null;
};
export const callLog: CallRecord[] = [];
export const clock = { t0: Date.now(), turn: null as number | null };

export type ChatMessage = { role: string; content: string | null; tool_calls?: unknown[]; tool_call_id?: string };
export type ChatReply = { text: string; toolCalls: Array<{ id: string; name: string; args: string }>; record: CallRecord };

/** One POST to the coding endpoint. Throws on HTTP/transport failure (the record is still logged). */
export async function chat(purpose: string, messages: ChatMessage[], opts: {
  maxTokens?: number; temperature?: number; tools?: unknown[];
} = {}): Promise<ChatReply> {
  const body: Record<string, unknown> = { model: ZAI_MODEL, messages };
  if (opts.maxTokens) body.max_tokens = Math.floor(opts.maxTokens);
  if (typeof opts.temperature === "number") body.temperature = opts.temperature;
  if (opts.tools?.length) body.tools = opts.tools;
  const rec: CallRecord = {
    purpose, t_start_ms: Date.now() - clock.t0, latency_ms: 0, model_readback: null,
    max_tokens: opts.maxTokens ?? null, prompt_chars: messages.reduce((n, m) => n + (m.content?.length ?? 0), 0),
    reply_chars: 0, usage: null, finish_reason: null, tool_calls: 0, error: null, turn: clock.turn,
  };
  const t = performance.now();
  try {
    const res = await fetch(ZAI_URL, {
      method: "POST", body: JSON.stringify(body), signal: AbortSignal.timeout(HTTP_TIMEOUT_MS),
      headers: { "Content-Type": "application/json", Authorization: `Bearer ${key()}` },
    });
    const raw = await res.text();
    rec.latency_ms = Math.round(performance.now() - t);
    if (!res.ok) throw new Error(`HTTP ${res.status}: ${raw.slice(0, 300)}`);
    const data = JSON.parse(raw);
    const choice = data?.choices?.[0] ?? {};
    const msg = choice.message ?? {};
    const toolCalls = (msg.tool_calls ?? []).map((c: any) => ({
      id: String(c.id), name: String(c.function?.name), args: String(c.function?.arguments ?? "{}"),
    }));
    Object.assign(rec, {
      model_readback: data?.model ?? null, usage: data?.usage ?? null, finish_reason: choice.finish_reason ?? null,
      reply_chars: (msg.content ?? "").length, tool_calls: toolCalls.length,
    });
    return { text: msg.content ?? "", toolCalls, record: rec };
  } catch (err) {
    rec.latency_ms ||= Math.round(performance.now() - t);
    rec.error = (err instanceof Error ? err.message : String(err)).slice(0, 300);
    throw err;
  } finally {
    callLog.push(rec);
  }
}

function textOf(content: unknown): string {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";
  return content.map((b: any) => (typeof b === "string" ? b : b?.type === "text" ? b.text ?? "" : textOf(b?.content)))
    .filter(Boolean).join("\n");
}

/** The injected summariser: {model, messages, system, maxTokens, temperature} -> {content:[{type:"text",text}]}. */
export const zaiComplete: CompleteFn = async (params) => {
  const messages: ChatMessage[] = [
    ...(params.system?.trim() ? [{ role: "system", content: params.system }] : []),
    ...params.messages.map((m) => ({ role: m.role === "assistant" ? "assistant" : m.role === "system" ? "system" : "user",
      content: textOf(m.content) })),
  ];
  try {
    const r = await chat("summariser", messages, { maxTokens: params.maxTokens, temperature: params.temperature });
    return { content: r.text ? [{ type: "text", text: r.text }] : [], provider: "zai", model: r.record.model_readback ?? ZAI_MODEL,
      usage: r.record.usage as Record<string, unknown> };
  } catch (err) {
    return { content: [], error: { kind: "provider_error", message: err instanceof Error ? err.message : String(err) } };
  }
};
