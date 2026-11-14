import { defineAgent } from "eve";
import { model } from "./lib/model";

export default defineAgent({
  model,
  modelContextWindowTokens: 30_000,
  compaction: { threshold: 0.6 },
});
