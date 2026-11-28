import { defineEval } from "eve/evals";
import { includes } from "eve/evals/expect";

export default defineEval({
  description: "Session B, new server process: find session A's facts with the LCM-X tools.",
  timeoutMs: 600_000,
  async test(t) {
    let turn = await t.send("In an earlier session I told you a project codename and a launch date. What were they?");
    t.succeeded();
    t.check(turn.reply ?? "", includes("BLUE-HERON-42"));
    t.check(turn.reply ?? "", includes("2026-12-03")).soft();
    turn = await turn.session.send("In that session you looked up the key gamma. What value did the lookup return?");
    t.check(turn.reply ?? "", includes("VALUE-gamma-7f3a"));
  },
});
