// S1 reader retry rule, the S2 rule (s2/run_s_lcmx.py probe()): at most 2 attempts per batch; the first answered
// value per probe wins (a retry only fills gaps); retry only when the attempt's final reader call hit the completion
// cap with probes unanswered; still unanswered after a capped attempt -> READER_TRUNCATED.
export const MAX_ATTEMPTS = 2;
export type RetryState = { answers: Record<string, unknown>; sources: Record<string, number>; capped: boolean;
  capHit: boolean; complete: boolean; attempts: number; stop: boolean; error?: string };

/** As score_s.py reads it: null/undefined or a blank string is unanswered; any other value is an answer. */
export const answered = (a: unknown) => a != null && (typeof a !== "string" || a.trim() !== "");

export function mergeAttempt(prev: RetryState | null, got: Record<string, unknown> | null, probeIds: string[],
  lastCompletionTokens: number, cap: number): RetryState {
  const answers = { ...prev?.answers }, sources = { ...prev?.sources }, attempts = (prev?.attempts ?? 0) + 1;
  for (const [id, a] of Object.entries(got ?? {})) if (!answered(answers[id])) {
    answers[id] = a;
    if (answered(a)) sources[id] = attempts; // the attempt whose reply supplied the scored answer
  }
  const capped = lastCompletionTokens >= cap, capHit = !!prev?.capHit || capped;
  const complete = probeIds.every((id) => answered(answers[id]));
  const stop = !capped || complete || attempts >= MAX_ATTEMPTS;
  return { answers, sources, capped, capHit, complete, attempts, stop,
    error: stop && capHit && !complete ? `READER_TRUNCATED: completion cap ${cap} reached` : undefined };
}

/** The final attempt's own failure (transport, timeout) outranks the synthesized truncation, as S2's exception path. */
export const retryError = (lastAttemptError: string | undefined, s: RetryState) => lastAttemptError ?? s.error;

/** What answer.json keeps so a merged score can be rebuilt: every attempt's raw reply, the merged answers, and the
 * attempt (1-based) that supplied each one. */
export const answerRecord = (attempts: { text?: string }[], s: RetryState) => ({
  attempt_replies: attempts.map((a) => a.text ?? ""), merged_answers: s.answers, answer_sources: s.sources });
