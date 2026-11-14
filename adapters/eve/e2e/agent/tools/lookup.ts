import { defineTool } from "eve/tools";
import { z } from "zod";

export default defineTool({
  description: "Look up the value stored for a key.",
  inputSchema: z.object({ key: z.string().min(1) }),
  async execute({ key }) {
    return { key, value: `VALUE-${key}-7f3a`, detail: `The record for ${key} was filed on 2026-10-02.` };
  },
});
