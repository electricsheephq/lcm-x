import { defineEval } from "eve/evals";
import { includes } from "eve/evals/expect";

const note = (index: number) => `Note ${index}: ` + `field report line ${index} `.repeat(600);

export default defineEval({
  description: "Capture through compaction, then recall an early tool result through lcmx__grep.",
  timeoutMs: 300_000,
  async test(t) {
    let turn = await t.send("lookup alpha");
    t.succeeded();
    t.check(turn.reply ?? "", includes("VALUE-alpha-7f3a"));
    for (let index = 0; index < 8; index += 1) {
      turn = await turn.session.send(note(index));
    }
    turn = await turn.session.send("lookup beta");
    t.check(turn.reply ?? "", includes("VALUE-beta-7f3a"));
    await new Promise((resolve) => setTimeout(resolve, 3_000)); // the writer runs after each capture
    turn = await turn.session.send("recall VALUE-alpha-7f3a");
    t.calledTool("lcmx__grep");
    t.check(turn.reply ?? "", includes("VALUE-alpha-7f3a"));
  },
});
