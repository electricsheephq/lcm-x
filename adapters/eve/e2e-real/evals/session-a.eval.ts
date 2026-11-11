import { defineEval } from "eve/evals";
import { includes } from "eve/evals/expect";

const report = (index: number) =>
  `Field report ${index}. Acknowledge it in one short sentence, nothing else.\n` +
  Array.from({ length: 160 }, (_, line) => `Report ${index}, line ${line}: sensor ${line % 7} read ${(line * 37) % 101} units at station ${index}.`).join("\n");

export default defineEval({
  description: "Session A: facts, a tool call and enough text that Eve compacts.",
  timeoutMs: 600_000,
  async test(t) {
    let turn = await t.send("Remember for later: the project codename is BLUE-HERON-42 and the launch date is 2026-12-03.");
    t.succeeded();
    turn = await turn.session.send("Please look up the key gamma.");
    t.calledTool("lookup");
    t.check(turn.reply ?? "", includes("VALUE-gamma-7f3a"));
    for (let index = 1; index <= 6; index += 1) turn = await turn.session.send(report(index));
    turn = await turn.session.send("Thanks. That is all for this session.");
    t.succeeded();
  },
});
