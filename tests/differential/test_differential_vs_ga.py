"""No differential imports or subprocesses until explicitly enabled."""
import os

import pytest

if os.environ.get("LCM_DIFFERENTIAL") != "1":
    pytest.skip("opt in with LCM_DIFFERENTIAL=1", allow_module_level=True)

from pathlib import Path

from .cases import exact_cases, generated_cases, projection_cases
from .run import run
from .trees import BASELINE_CONFIG, BASELINES, DEFAULT_BASELINE, PrerequisiteUnavailable, skip_or_fail


def test_default_baseline_is_v028_with_native_defaults():
    assert DEFAULT_BASELINE == "v0.28.0"
    assert BASELINES[DEFAULT_BASELINE] == "709099b3a397a7893a7abae8802ebe7bf605f79e"
    assert BASELINE_CONFIG[DEFAULT_BASELINE] == "bare"


@pytest.fixture(scope="module")
def outputs(tmp_path_factory):
    root = os.environ.get("LCM_DIFFERENTIAL_OUT")
    return Path(root) if root else tmp_path_factory.mktemp("differential")


def evaluate(cases, output):
    try:
        result = run(cases, output, os.environ.get("LCM_DIFFERENTIAL_BASELINE", DEFAULT_BASELINE),
                     os.environ.get("LCM_DIFFERENTIAL_CANDIDATE", "worktree"), int(os.environ.get("LCM_DIFFERENTIAL_SEED", "1227")))
    except PrerequisiteUnavailable as exc:
        skip_or_fail(str(exc))
    assert not result["unexplained"], f"unexplained differential reds: {output / 'summary.json'}"
    return result


@pytest.mark.parametrize("case", [case for case in exact_cases()
                                 if not case.get("absolute_xfail")
                                 and case["id"] != "i1225_objective_exceeds_room_32k"], ids=lambda case: case["id"])
def test_exact_shape(case, outputs):
    evaluate([case], outputs / case["id"])


def test_generated_sample(outputs):
    seed = int(os.environ.get("LCM_DIFFERENTIAL_SEED", "1227"))
    cases = generated_cases(int(os.environ.get("LCM_DIFFERENTIAL_CASES", "60")), seed)
    evaluate(cases, outputs / "generated")


def test_i1281_g0_keeps_newest_user_without_budget(outputs):
    case = generated_cases(1, 1227)[0]
    assert case["id"] == "g0"
    try:
        # v0.27.0 also loses these sentinels, so the historical R2 pin needs v0.26.2.
        result = run([case], outputs / "i1281_g0", "v0.26.2",
                     os.environ.get("LCM_DIFFERENTIAL_CANDIDATE", "worktree"), 1227)
    except PrerequisiteUnavailable as exc:
        skip_or_fail(str(exc))
    assert not result["unexplained"], result["unexplained"]


def test_projection(outputs):
    evaluate(projection_cases(int(os.environ.get("LCM_DIFFERENTIAL_PROJECTION_CASES", "1000")),
                              int(os.environ.get("LCM_DIFFERENTIAL_SEED", "1227"))), outputs / "projection")


def test_i1225_objective_exceeds_room_32k(outputs):
    case = next(case for case in exact_cases() if case["id"] == "i1225_objective_exceeds_room_32k")
    evaluate([case], outputs / case["id"])


def test_i1225_shape_fits(outputs):
    """This 22k-objective shape projects to 22,579 tokens and fits. It does not reproduce #1225's absolute
    failure, so it is pinned as a fit, not as a strict xfail; a reproducing shape needs the issue's host input."""
    import json
    path = outputs / "i1225_objective_exceeds_room_32k" / "summary.json"
    if not path.exists():
        case = next(case for case in exact_cases() if case["id"] == "i1225_objective_exceeds_room_32k")
        evaluate([case], path.parent)
    absolute_fit(json.loads(path.read_text()))


def absolute_fit(result):
    assert not result["absolute_over_window"], result["absolute_over_window"]


class ObjectiveOverflow(AssertionError):
    """The preserved objective leaves insufficient room for the newest turn."""


@pytest.mark.xfail(strict=True, raises=ObjectiveOverflow, reason="#1225: objective exceeds available window room")
def test_i1225_historical_objective_overflow(outputs):
    import json
    case = next(case for case in exact_cases() if case.get("absolute_xfail"))
    result = evaluate([case], outputs / case["id"])
    if result["absolute_over_window"]:
        ga = [json.loads(line) for line in (outputs / case["id"] / "public-ga.jsonl").read_text().splitlines()]
        assert ga[0]["calls"][0]["tokens"] > case["window"], "#1225 must also overflow on GA"
        raise ObjectiveOverflow(str(result["absolute_over_window"]))


from copy import deepcopy

from .driver import pair_errors
from .oracles import compare


def test_projection_delta_is_only_fit_with_ga_user_marks():
    case = dict(layer="projection", limit=100)
    ga = dict(out_tokens=90, user_marks=[0], input_mutated=False,
              user_rows={"0": dict(tokens=40, marked=True, preserved=False)})
    cand = dict(out_tokens=110, user_marks=[], input_mutated=False,
                user_rows={"0": dict(tokens=60, marked=False, preserved=True)})
    assert compare(case, ga, cand, baseline="v0.26.2")[0]["delta"] == "D916_GA_MARKED_USER"
    ga["user_marks"] = []
    ga["user_rows"] = {}
    assert compare(case, ga, cand, baseline="v0.26.2")[0]["delta"] is None
    cand.update(user_marks=[0], input_mutated=True)
    assert {row["oracle"] for row in compare(case, ga, cand, baseline="v0.26.2")} == {"R1", "P_USER", "P_INPUT"}


def test_public_oracles_and_delta_do_not_hide_absolute_failures():
    case = dict(layer="public", window=100, steps=[[]])
    call = dict(tokens=90, sentinels=["⟦objective⟧", "⟦newest⟧"], user_marks=[0], pair_errors=[],
                compress_runs=0, summarizer_calls=0)
    ga = dict(uncovered_rows=0, calls=[call], produced=["⟦r0.0⟧"], stored=["⟦r0.0⟧"])
    cand = deepcopy(ga)
    cand["calls"][0].update(tokens=110, sentinels=[], user_marks=[], pair_errors=["missing-result:c"],
                            compress_runs=3, summarizer_calls=3)
    cand["stored"] = []
    failures = compare(case, ga, cand, baseline="v0.26.2")
    assert {row["oracle"] for row in failures} == {"R1", "R2", "R3", "A1", "A2"}
    assert all(row["delta"] is None for row in failures if row["oracle"] != "R1")


@pytest.mark.parametrize("tokens,marks,oracle,delta_id", [
    (110, [], "R1", "D916_GA_MARKED_USER"),
    (90, [0], "P_USER", "D1233_LAST_RESORT_USER_MARK"),
])
def test_shipped_delta_is_not_absorbed_by_new_ga(tokens, marks, oracle, delta_id):
    case = dict(layer="projection", limit=100)
    ga = dict(out_tokens=90, user_marks=[0], input_mutated=False,
              user_rows={"0": dict(tokens=40, marked=True, preserved=False)})
    cand = dict(out_tokens=tokens, user_marks=marks, input_mutated=False,
                user_rows={"0": dict(tokens=40 if marks else 60, marked=bool(marks), preserved=not marks)})
    legacy = compare(case, ga, cand, baseline="v0.26.2")
    detail = [0] if oracle == "P_USER" else dict(ga=90, candidate=110, limit=100)
    assert legacy == [dict(oracle=oracle, detail=detail, delta=delta_id)]
    # The shipped user marks are inherited under the new GA, not an accepted delta.
    unabsorbed = [] if oracle == "P_USER" else [dict(legacy[0], delta=None)]
    assert compare(case, ga, cand, baseline="v0.27.0") == unabsorbed
    assert compare(case, ga, cand) == unabsorbed


@pytest.mark.parametrize("baseline", ["v0.26.2", "v0.27.0", "v0.28.0"])
@pytest.mark.parametrize("marks", [[2, 0], [2, 0, 1], [2]])
def test_projection_user_marks_are_relative_after_v026(baseline, marks):
    case = dict(layer="projection", limit=100)
    ga = dict(out_tokens=90, user_marks=[0, 2], input_mutated=False)
    cand = dict(ga, user_marks=marks)
    if baseline == "v0.26.2":
        delta = "D1233_LAST_RESORT_USER_MARK" if 1 not in marks else None
        expected = [dict(oracle="P_USER", detail=marks, delta=delta)]
    else:
        expected = [dict(oracle="P_USER", detail=[1], delta=None)] if 1 in marks else []
    assert compare(case, ga, cand, baseline=baseline) == expected


@pytest.mark.parametrize("baseline,config,inherited", [
    ("v0.26.2", "managed", dict(cases=0, marks=0)),
    ("v0.27.0", "bare", dict(cases=1, marks=1)),
    ("v0.28.0", "bare", dict(cases=1, marks=1)),
])
def test_baseline_defaults_and_inherited_user_marks(baseline, config, inherited, tmp_path, monkeypatch):
    import json
    monkeypatch.delenv("LCM_DIFFERENTIAL_BASELINE_CONFIG", raising=False)
    # p0 has the same one-row last-resort mark in both pinned trees.
    result = run(projection_cases(1, 1227), tmp_path / "run", baseline=baseline)
    assert result["baseline_config"] == config
    assert result["inherited_user_marks"] == inherited
    assert not result["unexplained"]
    assert json.loads((tmp_path / "run/summary.json").read_text())["inherited_user_marks"] == inherited
    assert f"Inherited user marks: {inherited['cases']} cases; {inherited['marks']} marks." in (tmp_path / "run/summary.md").read_text()


def test_tool_results_must_follow_calls_and_every_call_needs_a_result():
    assistant = {"role": "assistant", "tool_calls": [{"id": "c"}]}
    tool = {"role": "tool", "tool_call_id": "c", "content": "stub"}
    assert pair_errors([assistant, tool]) == []
    assert pair_errors([tool, assistant]) == ["result-before-call:c", "missing-result:c"]
    assert pair_errors([assistant]) == ["missing-result:c"]


class SummaryCoverageGap(AssertionError):
    """Stored history is outside both the live context and summary coverage."""


@pytest.mark.xfail(strict=True, raises=SummaryCoverageGap, reason="#1247: 32K history lacks summary coverage")
def test_i1247_32k_history_is_live_or_summarized(outputs):
    import json
    cid = "i1139_small_window_32k"
    path = outputs / cid / "summary.json"
    if not path.exists():
        evaluate([next(case for case in exact_cases() if case["id"] == cid)], path.parent)
    count = json.loads(path.read_text())["coverage"][cid]["candidate"]
    if count:
        raise SummaryCoverageGap(f"{count} stored rows neither live nor summarized")


def test_objective_is_checked_only_on_newest_user_row():
    from .driver import visible_sentinels
    first = dict(role="user", content="⟦r0⟧⟦objective⟧")
    later = dict(role="user", content="⟦r1⟧⟦newest⟧")
    assert "⟦objective⟧" in visible_sentinels([first], [first])
    assert "⟦objective⟧" not in visible_sentinels([later], [first])
    assert "⟦objective⟧" not in visible_sentinels([first, later], [first, later])
    later["content"] += "⟦objective⟧"
    assert "⟦objective⟧" in visible_sentinels([later], [first, later])


def test_managed_baseline_and_case_override_precedence():
    from .driver import MANAGED_BASELINE, config_overrides
    case = dict(overrides={})
    assert config_overrides(case, "bare") == {}
    assert config_overrides(case, "managed") == MANAGED_BASELINE
    assert MANAGED_BASELINE == dict(context_threshold=0.75, fresh_tail_count=24, fresh_tail_max_tokens=24000,
                                   leaf_chunk_tokens=8000, threshold_full_sweep_enabled=True,
                                   large_output_externalization_enabled=True, large_output_active_replay_stubbing_enabled=True)
    case["overrides"] = dict(leaf_chunk_tokens=123)
    assert config_overrides(case, "managed")["leaf_chunk_tokens"] == 123
    assert config_overrides(case, "bare") == case["overrides"]


def test_summary_coverage_counts_live_and_all_dag_message_sources():
    from types import SimpleNamespace
    from .driver import uncovered_rows
    rows = [dict(role="assistant", content=f"⟦r{i}⟧", store_id=i) for i in range(4)]
    nodes = [SimpleNamespace(source_type="messages", source_ids=[1, 2]),
             SimpleNamespace(source_type="summaries", source_ids=[3])]
    assert uncovered_rows(rows, [rows[0]], nodes) == 1
    assert uncovered_rows(rows, [rows[0], rows[3]], nodes) == 0
    assert uncovered_rows(rows, [], []) == 4


def test_c1_tolerance_and_direction():
    case = dict(id="coverage", layer="public", window=100, steps=[])
    ga = dict(calls=[], produced=[], stored=[], uncovered_rows=0)
    cand = dict(ga, uncovered_rows=2)
    assert compare(case, ga, cand) == []
    cand["uncovered_rows"] = 3
    assert compare(case, ga, cand) == [dict(oracle="C1", detail=dict(ga=0, candidate=3), delta=None)]
    assert compare(case, cand, ga) == []


def test_i1123_delta_does_not_absorb_other_cells():
    from .ACCEPTED_DELTAS import i1123_leaf_trade
    case = dict(id="i1139_small_window_64k")
    failure = dict(detail=dict(metric="summarizer_calls", ga=46, candidate=65))
    assert i1123_leaf_trade(case, {}, {}, failure)
    assert not i1123_leaf_trade(dict(id="i1139_small_window_48k"), {}, {}, failure)
    for detail in (dict(metric="compress_runs", ga=46, candidate=65),
                   dict(metric="summarizer_calls", ga=45, candidate=65)):
        assert not i1123_leaf_trade(case, {}, {}, dict(detail=detail))


def test_i1123_40k_delta_is_a_separate_exact_cell():
    from .ACCEPTED_DELTAS import i1123_40k_compress_trade
    case = dict(id="i1139_small_window_40k")
    failure = dict(detail=dict(metric="compress_runs", ga=93, candidate=113))
    assert i1123_40k_compress_trade(case, {}, {}, failure)
    assert not i1123_40k_compress_trade(dict(id="i1139_small_window_64k"), {}, {}, failure)
    for detail in (dict(metric="summarizer_calls", ga=93, candidate=113),
                   dict(metric="compress_runs", ga=92, candidate=113)):
        assert not i1123_40k_compress_trade(case, {}, {}, dict(detail=detail))


def test_i1233_delta_is_only_projection_user_marks_already_in_ga():
    from .ACCEPTED_DELTAS import last_resort_user_mark
    case = dict(layer="projection", limit=100)
    ga = dict(out_tokens=90, user_marks=[0, 2], input_mutated=False,
              user_rows={str(i): dict(tokens=40, marked=True, preserved=False) for i in (0, 2)})
    cand = dict(out_tokens=90, user_marks=[2, 0], input_mutated=False)
    assert compare(case, ga, cand, baseline="v0.26.2") == [dict(oracle="P_USER", detail=[2, 0], delta="D1233_LAST_RESORT_USER_MARK")]
    cand = dict(out_tokens=90, user_marks=[2], input_mutated=False,
                user_rows={"0": dict(tokens=60, marked=False, preserved=True),
                           "2": dict(tokens=40, marked=True, preserved=False)})
    assert compare(case, ga, cand, baseline="v0.26.2")[0]["delta"] == "D1233_LAST_RESORT_USER_MARK"
    cand["out_tokens"] = 101
    assert compare(case, ga, cand, baseline="v0.26.2") == [
        dict(oracle="R1", detail=dict(ga=90, candidate=101, limit=100), delta="D916_GA_MARKED_USER"),
        dict(oracle="P_USER", detail=[2], delta=None),
    ]
    cand.update(out_tokens=100, user_marks=[0, 2], user_rows=deepcopy(ga["user_rows"]))
    assert compare(case, ga, cand, baseline="v0.26.2")[0]["delta"] == "D1233_LAST_RESORT_USER_MARK"
    cand["out_tokens"] = 101
    failures = compare(case, ga, cand, baseline="v0.26.2")
    assert failures == [dict(oracle="R1", detail=dict(ga=90, candidate=101, limit=100), delta=None),
                        dict(oracle="P_USER", detail=[0, 2], delta=None)]
    ga["out_tokens"] = 101
    assert compare(case, ga, cand, baseline="v0.26.2")[0]["delta"] == "D1233_LAST_RESORT_USER_MARK"
    cand["user_marks"] = [1]
    assert compare(case, ga, cand, baseline="v0.26.2")[0]["delta"] is None
    cand["user_marks"] = []
    assert not last_resort_user_mark(case, ga, cand)
    assert compare(case, ga, cand, baseline="v0.26.2") == []


def projection_row_fixture():
    rows = [dict(role="user", content="first request", timestamp=10.0),
            dict(role="assistant", content="reply", timestamp=11.0),
            dict(role="user", content="second request", timestamp=12.0)]
    case = dict(layer="projection", limit=100, rows=[dict(message=row) for row in rows])
    record = dict(out=deepcopy(rows), out_tokens=90, user_marks=[], input_mutated=False,
                  user_sources={str(i): dict(store_id=i + 1, tokens=80, text=row["content"])
                                for i, row in enumerate(rows) if row["role"] == "user"})
    return case, record


@pytest.mark.parametrize("mutation", ["dropped", "truncated", "swapped", "replaced", "wrong-source"])
def test_f1244r_projection_requires_every_user_row_in_order(mutation):
    case, ga = projection_row_fixture()
    cand = deepcopy(ga)
    if mutation == "dropped":
        cand["out"].pop(0)
    elif mutation == "truncated":
        cand["out"][0]["content"] = "first"
    elif mutation == "swapped":
        cand["out"][0], cand["out"][2] = cand["out"][2], cand["out"][0]
    elif mutation == "replaced":
        cand["out"][0]["content"] = "other request"
    else:
        cand["out"][0]["content"] = (
            "[LCM survival fit: this user message (80 tokens) is stored verbatim as store id 3; "
            "lcm_expand / lcm_grep reach the full text. (projection 1200/600)]")
    failures = compare(case, ga, cand)
    assert [row["oracle"] for row in failures] == ["P_USER_ROWS"]
    extra = [] if mutation == "dropped" else [0]
    assert failures[0]["detail"] == dict(missing=[2] if mutation == "swapped" else [0], extra=extra)
    assert failures[0]["delta"] is None


@pytest.mark.parametrize("marked", [False, True])
def test_f1244r_projection_accepts_identical_or_same_row_mark(marked):
    case, ga = projection_row_fixture()
    cand = deepcopy(ga)
    if marked:
        cand["out"][0]["content"] = (
            "[LCM survival fit: this user message (80 tokens) is stored verbatim as store id 1; "
            "lcm_expand / lcm_grep reach the full text. (projection 1200/600)]")
        ga["user_marks"] = cand["user_marks"] = [0]
    assert compare(case, ga, cand) == []


@pytest.mark.parametrize("change", ["clean", "tracked", "staged", "untracked"])
def test_f1244r_worktree_identity_records_dirty_bytes(tmp_path, change):
    import hashlib
    import subprocess
    from .run import candidate_identity

    def git(*args):
        return subprocess.run(["git", "-C", str(tmp_path), *args], check=True, capture_output=True).stdout

    git("init", "-q")
    tracked = tmp_path / "source.py"
    tracked.write_text("original\n")
    git("add", "source.py")
    git("-c", "user.name=Test", "-c", "user.email=test@example.invalid", "-c", "commit.gpgsign=false", "commit", "-qm", "fixture")
    sha = git("rev-parse", "HEAD").decode().strip()
    assert candidate_identity(tmp_path) == dict(candidate=sha)
    if change == "clean":
        return
    path = tmp_path / "new.py" if change == "untracked" else tracked
    path.write_text("changed\n")
    if change == "staged":
        git("add", "source.py")
    diff = git("diff", "HEAD")
    expected_hash = hashlib.sha256(diff + (path.read_bytes() if change == "untracked" else b"")).hexdigest()
    result = candidate_identity(tmp_path)
    assert result == dict(candidate=f"{sha}+dirty",
                          candidate_dirty=dict(files=[path.name], diff_sha256=expected_hash))
    path.write_text("changed again\n")
    assert candidate_identity(tmp_path)["candidate_dirty"]["diff_sha256"] != expected_hash


@pytest.mark.parametrize("layer", ["projection", "public"])
@pytest.mark.parametrize("preserved,growth,overflow,matched", [
    (True, 20, 20, True), (True, 20, 21, False), (False, 20, 10, False), (True, 0, 10, False),
])
def test_f1244r_d916_only_absorbs_preserved_user_growth(layer, preserved, growth, overflow, matched):
    from .ACCEPTED_DELTAS import ga_marked_user
    case = dict(layer=layer, limit=100, window=100)
    ga = dict(tokens=90, out_tokens=90, user_marks=[0],
              user_rows={"0": dict(tokens=40, marked=True, preserved=False)})
    cand = dict(tokens=100 + overflow, out_tokens=100 + overflow, user_marks=[] if preserved else [0],
                user_rows={"0": dict(tokens=40 + growth, marked=not preserved, preserved=preserved)})
    a, b = (ga, cand) if layer == "projection" else (dict(calls=[ga]), dict(calls=[cand]))
    failure = dict(detail=dict(call=0)) if layer == "public" else None
    assert ga_marked_user(case, a, b, failure) is matched


def test_f1244r_d916_public_checks_only_the_offending_call():
    from .ACCEPTED_DELTAS import ga_marked_user
    case = dict(layer="public", window=100)
    ga_call = dict(tokens=90, user_marks=[0], user_rows={"0": dict(tokens=40, marked=True, preserved=False)})
    cand_call = dict(tokens=110, user_marks=[], user_rows={"0": dict(tokens=60, marked=False, preserved=True)})
    ga, cand = dict(calls=[ga_call, ga_call]), dict(calls=[cand_call, dict(ga_call, tokens=110)])
    assert ga_marked_user(case, ga, cand, dict(detail=dict(call=0)))
    assert not ga_marked_user(case, ga, cand, dict(detail=dict(call=1)))
    assert not ga_marked_user(case, ga, cand)


def test_f1244r_pair_errors_rejects_duplicate_results():
    assistant = dict(role="assistant", tool_calls=[dict(id="c")])
    tool = dict(role="tool", tool_call_id="c", content="result")
    assert pair_errors([assistant, tool]) == []
    assert pair_errors([assistant, tool, deepcopy(tool)]) == ["duplicate-result:c"]


@pytest.mark.parametrize("layer", ["projection", "public"])
def test_f1244r_same_user_mark_leaves_assistant_overflow_unexplained(layer):
    case = dict(layer=layer, limit=100, window=100, steps=[[]])
    ga_call = dict(out_tokens=90, tokens=90, user_marks=[0], input_mutated=False,
                   user_rows={"0": dict(tokens=40, marked=True, preserved=False)},
                   sentinels=[], pair_errors=[], compress_runs=0, summarizer_calls=0)
    cand_call = dict(ga_call, out_tokens=110, tokens=110)
    ga, cand = (ga_call, cand_call) if layer == "projection" else (
        dict(calls=[ga_call], produced=[], stored=[], uncovered_rows=0),
        dict(calls=[cand_call], produced=[], stored=[], uncovered_rows=0))
    r1 = [row for row in compare(case, ga, cand, baseline="v0.26.2") if row["oracle"] == "R1"]
    assert len(r1) == 1
    assert r1[0]["delta"] is None


@pytest.mark.parametrize("head,tail,tampered", [(8, 4, False), (4, 2, False), (4, 2, True)])
def test_f1244r_projection_verifies_marked_head_and_tail(head, tail, tampered):
    case, ga = projection_row_fixture()
    cand = deepcopy(ga)
    text = cand["out"][0]["content"]
    marker = ("[LCM survival fit: this user message (80 tokens) is stored verbatim as store id 1; "
              f"lcm_expand / lcm_grep reach the full text. (projection {head}/{tail})]")
    content = f"{text[:head]}\n...\n{marker}"
    if len(text) > 2 * head:
        content += f"\n...\n{text[-tail:]}"
    cand["out"][0]["content"] = "wrong" + content if tampered else content
    failures = compare(case, ga, cand)
    assert [row["oracle"] for row in failures] == (["P_USER_ROWS"] if tampered else [])


@pytest.mark.parametrize("marked", [False, True])
def test_f1244r_user_token_evidence_survives_removed_timestamps(marked):
    import re
    from types import SimpleNamespace
    from .driver import user_row_evidence
    source = dict(role="user", content="⟦rsource.0⟧ request", timestamp=10.0)
    content = ("[LCM survival fit: this user message (80 tokens) is stored verbatim as store id 1; "
               "lcm_expand / lcm_grep reach the full text. (projection 1200/600)]") if marked else source["content"]
    view = [dict(role="user", content=content)]
    pattern = re.compile(r"this (user) message \((\d+) tokens\).*store id (\d+);")
    store = SimpleNamespace(get=lambda sid: source if sid == 1 else None)
    evidence = user_row_evidence(view, [source], lambda row: len(row["content"]), lambda text: text, pattern, store)
    assert evidence == {"0": dict(tokens=len(content), marked=marked, preserved=not marked)}


@pytest.mark.parametrize("mutation", ["inserted", "duplicated", "clean"])
def test_f1244r2_projection_user_sequence_is_exact(mutation):
    case, ga = projection_row_fixture()
    cand = deepcopy(ga)
    if mutation != "clean":
        extra = dict(role="user", content="inserted") if mutation == "inserted" else deepcopy(cand["out"][0])
        cand["out"].insert(1, extra)
    failures = compare(case, ga, cand)
    assert failures == ([] if mutation == "clean" else [
        dict(oracle="P_USER_ROWS", detail=dict(missing=[], extra=[1]), delta=None)])


@pytest.mark.parametrize("shape,expected", [
    ("reversed", ["result-out-of-order:b"]),
    ("separated", ["result-not-contiguous:a"]),
    ("duplicate-call", ["duplicate-call:a"]),
    ("valid", []),
])
def test_f1244r2_provider_safe_tool_groups(shape, expected):
    assistant = dict(role="assistant", tool_calls=[dict(id="a"), dict(id="b")])
    a, b = [dict(role="tool", tool_call_id=cid, content="result") for cid in ("a", "b")]
    rows = [assistant, a, b]
    if shape == "reversed":
        rows = [assistant, b, a]
    elif shape == "separated":
        rows = [dict(role="assistant", tool_calls=[dict(id="a")]), dict(role="user", content="next"), a]
    elif shape == "duplicate-call":
        rows = [dict(role="assistant", tool_calls=[dict(id="a"), dict(id="a")]), a]
    assert pair_errors(rows) == expected


@pytest.mark.parametrize("side", ["candidate", "ga"])
def test_f1244r2_missing_override_seam(side, tmp_path):
    from types import SimpleNamespace
    from .driver import run_case

    config = SimpleNamespace()
    engine = SimpleNamespace(
        on_session_start=lambda *a, **kw: None, shutdown=lambda: None,
        _store=SimpleNamespace(append=lambda *a, **kw: 1),
        _survival_projection=lambda rows, ids, limit: rows, _survival_measure=lambda rows: 0)
    modules = dict(
        config=SimpleNamespace(LCMConfig=lambda **kw: config),
        engine=SimpleNamespace(LCMEngine=lambda **kw: engine),
        escalation=SimpleNamespace(_call_llm_for_summary=None),
        survival_fit=SimpleNamespace(_PROJECTED_RE=None),
        message_content=SimpleNamespace(normalize_content_value=lambda value: value),
        tokens=SimpleNamespace(count_message_tokens=lambda row: 0))
    case = dict(id="missing-override", layer="projection", summarizer="healthy", window=100,
                limit=100, rows=[], overrides=dict(unavailable_setting=True))
    result = (run_case(case, modules, tmp_path, []) if side == "candidate"
              else run_case(case, modules, tmp_path, [], side=side))
    assert "config.unavailable_setting" in result["missing_seams"]
    if side == "candidate":
        assert "unavailable_setting" in result["error"]
        assert [failure["oracle"] for failure in compare(case, {}, result)] == ["DRIVER"]
    else:
        assert "error" not in result
        assert result["out"] == []


@pytest.mark.parametrize("ref_form", ["tag", "full", "short"])
def test_f1244r2_baseline_cli_canonicalizes_known_commit(ref_form, tmp_path, monkeypatch, capsys):
    import json
    import sys
    from .run import main
    from .trees import BASELINES

    monkeypatch.delenv("LCM_DIFFERENTIAL_BASELINE_CONFIG", raising=False)
    tag = "v0.26.2"
    ref = {"tag": tag, "full": BASELINES[tag], "short": BASELINES[tag][:9]}[ref_form]
    out = tmp_path / ref_form
    monkeypatch.setattr(sys, "argv", ["differential", "--baseline", ref, "--layers", "projection",
                                   "--cases", "1", "--seed", "1227", "--out", str(out)])
    assert main() == 0
    result = json.loads((out / "summary.json").read_text())
    assert result["baseline_config"] == "managed"
    assert result["counts"] == {"projection/p34/P_USER": 1}
    assert result["accepted"][0]["delta"] == "D1233_LAST_RESORT_USER_MARK"
    assert not result["unexplained"]
    assert result["baseline_ref"] == ref
    assert result["baseline_tag"] == tag
    assert "0 unexplained events" in capsys.readouterr().out


@pytest.mark.parametrize("shape,expected", [
    ("complete", []),
    ("duplicate-call", ["duplicate-call:c"]),
    ("duplicate-result", ["duplicate-result:c"]),
    ("missing-first", ["missing-result:c"]),
    ("missing-second", ["missing-result:c"]),
])
def test_f1244r3_reused_ids_are_scoped_to_call_group(shape, expected):
    assistant = dict(role="assistant", tool_calls=[dict(id="c")])
    tool = dict(role="tool", tool_call_id="c", content="result")
    second = deepcopy(assistant)
    if shape == "duplicate-call":
        second["tool_calls"].append(dict(id="c"))
    rows = [assistant, tool, second, deepcopy(tool)]
    if shape == "duplicate-result":
        rows.append(deepcopy(tool))
    elif shape == "missing-first":
        rows.pop(1)
    elif shape == "missing-second":
        rows.pop()
    assert pair_errors(rows) == expected


def test_f1244r3_cli_identity_precedes_in_repo_output(tmp_path, monkeypatch):
    import json
    import subprocess
    import sys
    from . import run as runner

    def git(*args):
        return subprocess.run(["git", "-C", str(tmp_path), *args], check=True, capture_output=True).stdout

    git("init", "-q")
    (tmp_path / "source.py").write_text("original\n")
    git("add", "source.py")
    git("-c", "user.name=Test", "-c", "user.email=test@example.invalid", "-c", "commit.gpgsign=false", "commit", "-qm", "fixture")
    sha = git("rev-parse", "HEAD").decode().strip()
    out = tmp_path / "runs" / "result"

    def materialize(ref, cache):
        tree = cache / runner.resolve(ref)
        tree.mkdir(parents=True)
        (tree / "source.py").write_text("baseline cache\n")
        return tree

    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "materialize", materialize)
    monkeypatch.setattr(sys, "argv", ["differential", "--layers", "projection", "--cases", "0", "--out", str(out)])
    assert runner.main() == 0
    assert (out.parent / "trees").is_dir()
    summary = json.loads((out / "summary.json").read_text())
    assert summary["candidate"] == sha
    assert "candidate_dirty" not in summary
    assert runner.candidate_identity(tmp_path)["candidate"] == f"{sha}+dirty"
