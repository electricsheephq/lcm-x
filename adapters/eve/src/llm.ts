import { streamText, type LanguageModel, type ModelMessage } from "ai";

export type ReasoningEffort = "minimal" | "low" | "medium" | "high";

export interface LlmBridgeOptions {
  /** The agent's own model: the same value `agent.ts` uses, such as `chatgpt("gpt-6.1-terra")`. */
  model?: LanguageModel;
  /** A model for LCM-X's calls instead of `model` (Hermes: the auxiliary model). */
  summaryModel?: LanguageModel;
}

interface CompleteParams {
  task?: string;
  messages?: { role?: string; content?: unknown }[];
  max_tokens?: number | null;
  reasoning_effort?: string | null;
}

const EFFORTS = new Set(["minimal", "low", "medium", "high"]);

/**
 * The Eve side of LCM-X's `call_llm`: one stateless model call with only LCM-X's own messages (its summarizer or
 * expansion prompt). No agent history, no tools, nothing in the agent's session. One call at a time.
 */
export function createLlmHandler(options: LlmBridgeOptions, timeoutMs: number) {
  const model = options.summaryModel ?? options.model;
  let queue: Promise<unknown> = Promise.resolve();

  async function complete(params: CompleteParams): Promise<unknown> {
    if (!model) throw new Error("no model: pass `model` (the agent's model) to defineLcmxMemory");
    // AI SDK 7 takes system text as `instructions`, not as a message.
    const text = (content: unknown) => (typeof content === "string" ? content : JSON.stringify(content ?? ""));
    const instructions = (params.messages ?? []).filter((message) => message.role === "system").map((message) => text(message.content));
    const messages: ModelMessage[] = (params.messages ?? [])
      .filter((message) => message.role !== "system")
      .map((message) => ({ role: message.role === "assistant" ? "assistant" : "user", content: text(message.content) }) as ModelMessage);
    const effort = typeof params.reasoning_effort === "string" && EFFORTS.has(params.reasoning_effort) ? params.reasoning_effort : undefined;
    // Streamed: the ChatGPT subscription backend refuses a non-streamed request ("Stream must be set to true"); Eve's
    // own agent loop streams too. The text is collected and returned in one piece.
    let failure: unknown;
    const result = streamText({
      model,
      ...(instructions.length ? { instructions: instructions.join("\n\n") } : {}),
      messages,
      ...(params.max_tokens ? { maxOutputTokens: params.max_tokens } : {}),
      ...(effort ? { providerOptions: { openai: { reasoningEffort: effort } } } : {}),
      abortSignal: AbortSignal.timeout(timeoutMs),
      onError: ({ error }) => {
        failure = error;
      },
    });
    let content: string;
    try {
      content = await result.text;
    } catch (error) {
      throw describe(failure ?? error);
    }
    if (failure) throw describe(failure);
    const [finishReason, usage, response] = await Promise.all([result.finishReason, result.usage, result.response]);
    return {
      content,
      finish_reason: finishReason === "length" ? "length" : finishReason === "content-filter" ? "content_filter" : "stop",
      model: response?.modelId,
      usage: {
        prompt_tokens: usage?.inputTokens ?? 0,
        completion_tokens: usage?.outputTokens ?? 0,
        total_tokens: usage?.totalTokens ?? 0,
      },
    };
  }

  return {
    enabled: model !== undefined,
    handle(method: string, params: Record<string, unknown>): Promise<unknown> {
      if (method !== "llm/complete") return Promise.reject(new Error(`unknown host request: ${method}`));
      const run = queue.then(() => complete(params as CompleteParams));
      queue = run.catch(() => undefined); // low priority and never parallel to itself
      return run;
    },
  };
}

/** The model's error with its HTTP status and the backend's answer; never the request (it holds transcript text). */
function describe(error: unknown): Error {
  const detail = error as { message?: string; statusCode?: number; responseBody?: string } | undefined;
  const status = detail?.statusCode ? ` (HTTP ${detail.statusCode})` : "";
  const body = detail?.responseBody ? `: ${String(detail.responseBody).slice(0, 300)}` : "";
  return new Error(`${detail?.message ?? String(error)}${status}${body}`);
}

export type EmbeddingsOption =
  | "off"
  | "fastembed"
  | {
      provider: "openai";
      /** Default `text-embedding-3-small`. */
      model?: string;
      /** Variable that holds the API key. Default `OPENAI_API_KEY`. The key is used for embeddings only. */
      apiKeyEnv?: string;
      /** Stored dimensions (Matryoshka cut). Default 512. */
      storeDim?: number;
      /** Default `https://api.openai.com/v1`. */
      baseUrl?: string;
    };

/** LCM-X settings for the host's environment. */
export function hostEnv(embeddings: EmbeddingsOption, summaryEffort: ReasoningEffort): Record<string, string> {
  const env: Record<string, string> = {
    LCM_SUMMARY_REASONING_EFFORT: summaryEffort,
    LCM_EXPANSION_REASONING_EFFORT: summaryEffort,
  };
  if (embeddings === "off") return { ...env, LCM_EMBEDDINGS_ENABLED: "false" };
  if (embeddings === "fastembed") return { ...env, LCM_EMBEDDINGS_ENABLED: "true", LCM_EMBEDDING_PROVIDER: "fastembed" };
  return {
    ...env,
    LCM_EMBEDDINGS_ENABLED: "true",
    LCM_EMBEDDING_PROVIDER: "openai-compatible",
    LCM_EMBEDDING_MODEL: embeddings.model ?? "text-embedding-3-small",
    LCM_EMBEDDING_BASE_URL: embeddings.baseUrl ?? "https://api.openai.com/v1",
    LCM_EMBEDDING_API_KEY_ENV: embeddings.apiKeyEnv ?? "OPENAI_API_KEY",
    LCM_EMBEDDING_STORE_DIM: String(embeddings.storeDim ?? 512),
    LCM_EMBEDDING_STORAGE_DTYPE: "int8",
    LCM_EMBEDDING_BINARY_PRESCREEN: "true",
  };
}
