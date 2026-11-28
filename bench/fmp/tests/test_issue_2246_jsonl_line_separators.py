"""REGRESSION (#2246): bench/fmp JSONL readers split records on "\\n" only.

Codex writes U+2028, U+2029 and U+0085 raw inside JSON strings; ``str.splitlines()`` cut such a
record mid-string, and the derived cache's event trace dropped the answer silently.
"""
import json
import subprocess

from bench.fmp import derived_cache as cache
from bench.fmp import models

TEXT = "first second third\x85fourth"
EVENTS = [{"type": "thread.started", "thread_id": "thread"},
          {"type": "item.completed", "item": {"type": "agent_message", "text": TEXT}},
          {"type": "turn.completed"}]


def test_codex_answer_with_line_separators_survives_the_event_trace():
    target = models.Codex.__new__(models.Codex)  # no harness directories: only the runner wrapper is exercised
    stdout = "\n".join(json.dumps(event, ensure_ascii=False) for event in EVENTS) + "\n"
    target.run = lambda *args, **kwargs: subprocess.CompletedProcess([], 0, stdout, "")
    target.call = lambda spec, messages, records, **kwargs: target.run()
    proc = cache.backend_call(target, None, [], [])
    assert [json.loads(line) for line in proc.stdout.split("\n")] == EVENTS


def test_check_arms_reads_answers_with_raw_line_separators(tmp_path):
    from bench.fmp import run
    from bench.fmp.tests.test_995 import make_answers
    arms = [make_answers(tmp_path, name, label) for name, label in (("A0", "role"), ("A1", "speaker"))]
    for directory in arms:  # answers.jsonl is written with ensure_ascii=False (run.py), so separators stay raw
        row = json.loads((directory / "answers.jsonl").read_text())
        (directory / "answers.jsonl").write_text(json.dumps({**row, "hypothesis": TEXT}, ensure_ascii=False) + "\n")
    report = run.check_arms(arms, tmp_path / "check.json")
    assert report["passed"] is True
