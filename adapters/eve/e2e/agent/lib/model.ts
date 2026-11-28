import { mockModel } from "eve/evals";

// The fixture agent's model, shared by agent.ts and the LCM-X memory slot (LCM-X's summaries run on it, as Hermes
// runs them on the agent's auxiliary model). It answers LCM-X's summary contract when it sees one.
export const model = mockModel({
  modelId: "lcmx-e2e",
  respond: ({ messages, lastUserMessage, toolResults }) => {
    const all = messages.map((message) => message.text).join("\n");
    const nonce = /<lcm-summary nonce="([0-9a-f]+)">/.exec(all)?.[1];
    if (nonce) {
      return `<lcm-summary nonce="${nonce}">\nThe user stored field notes and looked up the key alpha (VALUE-alpha-7f3a).\nExpand for details about: the notes and lookups\n</lcm-summary>`;
    }
    if (all.includes("CONTEXT CHECKPOINT COMPACTION")) return "Checkpoint: the user stored notes and looked up keys.";
    const last = messages.at(-1);
    const text = lastUserMessage ?? "";
    if (last?.role === "tool") {
      const output = JSON.stringify(toolResults.at(-1)?.output ?? null);
      return `Tool said: ${output.slice(0, 400)}`;
    }
    if (text.startsWith("lookup ")) return { toolCalls: [{ name: "lookup", input: { key: text.slice(7).trim() } }] };
    if (text.startsWith("recall ")) return { toolCalls: [{ name: "lcmx__grep", input: { query: text.slice(7).trim() } }] };
    return `Noted (${text.length} chars).`;
  },
});

