import { createHash } from "node:crypto";

/** A JSON value with object keys in sorted order. */
export type Json = null | boolean | number | string | Json[] | { [key: string]: Json };

/**
 * Eve history messages as plain, canonical JSON: keys sorted, binary data as `{ $bytes: base64 }`,
 * URLs as `{ $url }`, dates as ISO text, `undefined` and functions left out. The spool keeps this form,
 * and the digest is taken over it, so the same message always has the same digest.
 */
export function normalize(value: unknown): Json {
  if (value === null || typeof value === "string" || typeof value === "boolean") return value;
  if (typeof value === "number") return Number.isFinite(value) ? value : String(value);
  if (typeof value === "bigint") return value.toString();
  if (value === undefined || typeof value === "function" || typeof value === "symbol") return null;
  if (value instanceof Uint8Array) return { $bytes: Buffer.from(value.buffer, value.byteOffset, value.byteLength).toString("base64") };
  if (value instanceof ArrayBuffer) return { $bytes: Buffer.from(value).toString("base64") };
  if (value instanceof URL) return { $url: value.href };
  if (value instanceof Date) return value.toISOString();
  if (Array.isArray(value)) return value.map((item) => normalize(item));
  const out: { [key: string]: Json } = {};
  for (const key of Object.keys(value as object).sort()) {
    const item = (value as Record<string, unknown>)[key];
    if (item === undefined || typeof item === "function" || typeof item === "symbol") continue;
    out[key] = normalize(item);
  }
  return out;
}

export function sha256(text: string): string {
  return createHash("sha256").update(text, "utf8").digest("hex");
}

/** Digest of one normalized message. */
export function digest(message: Json): string {
  return sha256(JSON.stringify(message));
}

/** Digest of a history prefix: sha256 over the message digests joined by newlines (the writer checks it). */
export function chain(digests: readonly string[]): string {
  return sha256(digests.join("\n"));
}
