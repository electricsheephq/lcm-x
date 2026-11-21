import { defineAgent } from "eve";
import { openai } from "eve/models/openai";

// Direct OpenAI platform key (OPENAI_API_KEY), as Tealbrick uses. A small window makes Eve compact in session A.
export default defineAgent({
  model: openai("gpt-6-luna"), // Eve's default gpt-6-luna-fast is not on every platform key
  modelContextWindowTokens: 24_000,
  compaction: { threshold: 0.6 },
});
