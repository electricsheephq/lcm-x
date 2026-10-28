"""Explicit model-free wiring lane; deliberately says nothing about retention quality."""
import json
import re
from types import SimpleNamespace


class SummaryLane:
    fake = True

    def call(self, system, user, max_tokens):
        body = "Offline fixture replay preserves the chronological session and its pending request. This is a model-free wiring summary.\nExpand for details about: the fixture replay."
        tag = re.search(r'<lcm-summary nonce="[a-f0-9]+">', system or "")
        text = tag[0] + "\n" + body + "\n</lcm-summary>" if tag else body
        return SimpleNamespace(text=text, meta=dict(prompt_tokens=(len(system or "") + sum(len(x) for x in user)) // 4,
                               completion_tokens=len(text) // 4, lane_model="FAKE", latency_s=0.0))


class Reader:
    readback = dict(model="FAKE", pin_ok=True)

    def ask(self, system, view, turns, replies):
        ids = re.findall(r"^([^\s:]+): ", turns[-1], re.M)
        answers = dict.fromkeys(ids, "I don't know.")
        for pid in re.findall(r"^([^\s:]+): .*Begin your answer with exactly one line:", turns[-1], re.M):
            answers[pid] = "STATUS: LIVE\nOffline fixture; no recall claim."
        return json.dumps(answers), dict(context_format="FAKE",
            usage=dict(prompt_tokens=(len(system) + sum(len(str(m.get('content', ''))) for m in view)
                                      + len(turns[-1])) // 4, completion_tokens=len(ids) * 6,
                       cached_tokens=0, uncached_tokens=(len(system) + sum(len(str(m.get('content', ''))) for m in view)
                                                        + len(turns[-1])) // 4, latency_s=0.0))
