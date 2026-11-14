import { spawn, type ChildProcess } from "node:child_process";
import { mkdir } from "node:fs/promises";
import { join } from "node:path";

export interface RuntimePaths {
  /** Absolute Python 3.11+ interpreter whose SQLite is 3.51.3 or newer. */
  python: string;
  /** Absolute LCM-X core: the package's vendored `core/`, or a checkout of this repository. */
  packageRoot: string;
  /** Absolute engine host script. Default: `<packageRoot>/adapters/eve/host/lcmx_eve_host.py` (a checkout). */
  hostScript?: string;
}

export interface HostSettings {
  /** The context window the engine compacts its own view of a session for (Hermes: the model's context length). */
  contextLength: number;
  /** Model name the engine records (Hermes passes the session's model). */
  modelName: string;
  /** Extra environment for the host (LCM-X settings: embeddings, reasoning effort). */
  env: Record<string, string>;
  /**
   * Answers the host's requests (`llm/complete`: LCM-X's call_llm, run on the agent's model). When set, the host
   * is started with `--llm-bridge` and may write LLM summaries.
   */
  onRequest?: (method: string, params: Record<string, unknown>) => Promise<unknown>;
}

export function hostScript(paths: RuntimePaths): string {
  return paths.hostScript ?? join(paths.packageRoot, "adapters", "eve", "host", "lcmx_eve_host.py");
}

interface Pending {
  resolve: (value: unknown) => void;
  reject: (error: Error) => void;
  timer: NodeJS.Timeout;
}

/**
 * The LCM-X engine host (`lcmx_eve_host.py serve`) for one store home: a stdio child that runs the real
 * LCMEngine and speaks newline-delimited JSON-RPC. Started on first use and kept for the life of the Eve
 * process (the engine holds the store, as Hermes does); it ends when Eve's process ends (stdin closes).
 * Restarted on the next call after a crash.
 */
export class HostClient {
  private child: ChildProcess | null = null;
  private ready: Promise<void> | null = null;
  private buffer = "";
  private nextId = 1;
  private readonly pending = new Map<number, Pending>();

  constructor(
    private readonly runtime: () => RuntimePaths,
    private readonly settings: HostSettings,
    private readonly home: string,
    private readonly onProblem: (event: string, detail: Record<string, unknown>) => void,
  ) {}

  /** Ask the host to drain the spool now; never waits and never throws (the host also scans every 2 s). */
  nudge(): void {
    this.start()
      .then(() => this.request("drain", {}, 600_000))
      .catch((error) => this.onProblem("host_drain_failed", { home: this.home, error: String(error) }));
  }

  /** An Eve turn starts: the host holds back idle summaries until its turn.completed. Starts the host. */
  turnStarted(): void {
    this.start()
      .then(() => this.request("turn/started", {}, 120_000))
      .catch((error) => this.onProblem("host_turn_failed", { home: this.home, error: String(error) }));
  }

  /** Drain the spool and wait for the report (tests, shutdown). */
  async drain(timeoutMs = 600_000): Promise<unknown> {
    await this.start();
    return this.request("drain", {}, timeoutMs);
  }

  async callTool(session: string, name: string, args: Record<string, unknown>, timeoutMs: number): Promise<string> {
    await this.start();
    const result = (await this.request("tools/call", { session, name, arguments: args }, timeoutMs)) as {
      content?: { type: string; text?: string }[];
    };
    return (result.content ?? []).map((part) => part.text ?? "").join("");
  }

  stop(): void {
    const child = this.child;
    this.child = null;
    this.ready = null;
    for (const [, waiter] of this.pending) {
      clearTimeout(waiter.timer);
      waiter.reject(new Error("host stopped"));
    }
    this.pending.clear();
    if (child && child.exitCode === null) child.stdin?.end(); // the host drains, closes the engine and exits
  }

  private start(): Promise<void> {
    if (this.ready) return this.ready;
    this.ready = (async () => {
      await mkdir(join(this.home, "logs"), { recursive: true, mode: 0o700 });
      const paths = this.runtime();
      const child = spawn(
        paths.python,
        ["-I", hostScript(paths), "serve", "--home", this.home, "--core", paths.packageRoot,
          "--context-length", String(this.settings.contextLength), "--model", this.settings.modelName,
          ...(this.settings.onRequest ? ["--llm-bridge"] : [])],
        { cwd: this.home, stdio: ["pipe", "pipe", "ignore"], env: { ...process.env, ...this.settings.env } },
      );
      this.child = child;
      child.stdout!.setEncoding("utf8");
      child.stdout!.on("data", (chunk: string) => this.onData(chunk));
      child.on("exit", (code) => {
        if (code) this.onProblem("host_exited", { home: this.home, code });
        if (this.child === child) this.stop();
      });
      child.on("error", (error) => {
        this.onProblem("host_spawn_failed", { home: this.home, error: String(error) });
        if (this.child === child) this.stop();
      });
      // The host must not keep Eve's process alive on its own.
      child.unref();
      (child.stdout as unknown as { unref?: () => void }).unref?.();
      (child.stdin as unknown as { unref?: () => void }).unref?.();
      await this.request("initialize", { client: "lcmx-eve" }, 120_000);
    })();
    this.ready.catch(() => this.stop());
    return this.ready;
  }

  private send(message: unknown): void {
    this.child?.stdin?.write(JSON.stringify(message) + "\n");
  }

  private request(method: string, params: unknown, timeoutMs: number): Promise<unknown> {
    const id = this.nextId++;
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        this.pending.delete(id);
        reject(new Error(`LCM-X host ${method} timed out after ${timeoutMs} ms`));
      }, timeoutMs);
      timer.unref();
      this.pending.set(id, { resolve, reject, timer });
      if (!this.child?.stdin?.writable) {
        clearTimeout(timer);
        this.pending.delete(id);
        reject(new Error("LCM-X host is not running"));
        return;
      }
      this.send({ jsonrpc: "2.0", id, method, params });
    });
  }

  private onData(chunk: string): void {
    this.buffer += chunk;
    let newline: number;
    while ((newline = this.buffer.indexOf("\n")) >= 0) {
      const line = this.buffer.slice(0, newline).trim();
      this.buffer = this.buffer.slice(newline + 1);
      if (!line) continue;
      let message: { id?: number | string; method?: string; params?: Record<string, unknown>; result?: unknown; error?: { message?: string } };
      try {
        message = JSON.parse(line);
      } catch {
        continue;
      }
      if (typeof message.method === "string" && message.id !== undefined) {
        this.answer(message.id, message.method, message.params ?? {});
        continue;
      }
      const waiter = typeof message.id === "number" ? this.pending.get(message.id) : undefined;
      if (!waiter) continue;
      this.pending.delete(message.id as number);
      clearTimeout(waiter.timer);
      if (message.error) waiter.reject(new Error(message.error.message ?? "host error"));
      else waiter.resolve(message.result);
    }
  }

  /** A request from the host (the LLM bridge). Never throws: the host gets a result or an error. */
  private answer(id: number | string, method: string, params: Record<string, unknown>): void {
    const handler = this.settings.onRequest;
    const work = handler ? handler(method, params) : Promise.reject(new Error(`no handler for ${method}`));
    work.then(
      (result) => this.send({ jsonrpc: "2.0", id, result }),
      (error) => this.send({ jsonrpc: "2.0", id, error: { code: -32000, message: error instanceof Error ? error.message : String(error) } }),
    );
  }
}
