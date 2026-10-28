"""Replayable synthetic cases. Randomness is confined to generation."""
from contextlib import contextmanager
from copy import deepcopy
import importlib.util
import json
import math
from pathlib import Path
import random
import sys

from .driver import module_patches

PROSE = "The record shows a value that we compare with earlier evidence before the next step. "
LENS = [0, 1, 50, 300, 900, 1100, 1199, 1200, 1201, 1500, 2000, 2399, 2400, 2401, 3000, 4500, 6000]
FLEET = dict(leaf_chunk_tokens=8000, threshold_full_sweep_enabled=True,
             large_output_externalization_enabled=True, large_output_active_replay_stubbing_enabled=True,
             large_output_externalization_threshold_chars=12000,
             large_output_active_replay_stub_threshold_tokens=10000,
             large_output_active_replay_stub_aged_threshold_tokens=2000)


def body(n):
    return (PROSE * (n // len(PROSE) + 1))[:n]


def call(cid):
    return {"id": cid, "type": "function", "function": {"name": "probe", "arguments": "{}"}}


def pairs(n, assistant=40, result=1199, prefix="c"):
    return [row for i in range(n) for row in (
        {"role": "assistant", "content": body(assistant), "tool_calls": [call(f"{prefix}{i}")]},
        {"role": "tool", "content": body(result), "tool_call_id": f"{prefix}{i}", "name": "probe", "tool_name": "probe"},
    )]


def tagged(case):
    """Keep boundary-length prose at its requested length where a tag fits."""
    rows = [row for step in case["steps"] for row in step]
    users = [i for i, row in enumerate(rows) if row["role"] == "user"]
    for i, row in enumerate(rows):
        tag = f"⟦r{case['id']}.{i}⟧"
        if users and i == users[0]:
            tag += "⟦objective⟧"
        if users and i == users[-1]:
            tag += "⟦newest⟧"
        value = row.get("content", "")
        row["content"] = tag + value[len(tag):]
        row["timestamp"] = float(i + 1)
    return case


def public_case(cid, rows, **kw):
    case = dict(id=cid, family=cid.split("_")[0], layer="public", window=32768,
                config="defaults", overrides={}, summarizer="healthy", steps=[rows],
                call_seconds=15, summary_tokens=80)
    case.update(kw)
    if case["config"] == "fleet":
        case["overrides"] = {**FLEET, **case["overrides"]}
    return tagged(case)


def exact_cases():
    cases = [public_case(f"i1227_seq_outage_n{n}", [{"role": "user", "content": "Check records."},
                         *pairs(n, 1100, 2)], summarizer="outage") for n in (90, 113, 128)]
    rows = [{"role": "user", "content": "Earlier question."}, {"role": "assistant", "content": "Earlier answer."},
            {"role": "user", "content": "Run the batch."},
            {"role": "assistant", "content": "", "tool_calls": [call(f"c{i}") for i in range(128)]},
            *[{"role": "tool", "content": body(1199), "tool_call_id": f"c{i}", "name": "probe", "tool_name": "probe"}
              for i in range(128)]]
    cases.append(public_case("i1220_parallel_fleet_n128", rows, config="fleet"))
    history = [{"role": "system", "content": "Synthetic system prompt."},
               {"role": "user", "content": "Earlier request."}, {"role": "assistant", "content": "Earlier answer."},
               {"role": "user", "content": "Keep this objective verbatim. " + "plan " * 9600},
               {"role": "assistant", "content": "Starting work."}]
    case = public_case("i1221_objective_default_32k", history + pairs(10, 8, 7509), leaf_history=4)
    # The objective is the last user request, not the earlier conversational preamble.
    case["steps"][0][1]["content"] = case["steps"][0][1]["content"].replace("⟦objective⟧", "")
    case["steps"][0][3]["content"] = case["steps"][0][3]["content"].replace("⟦newest⟧", "⟦objective⟧⟦newest⟧")
    cases.append(case)
    for window in (32000, 40000, 48000, 64000, 128000):
        steps = []
        for turn in range(40):
            for model_call in range(9):
                rows = pairs(1, 800, 8000, f"t{turn}c{model_call}")
                if model_call == 0:
                    rows.insert(0, {"role": "user", "content": f"Continue record {turn}."})
                steps.append(rows)
        cases.append(public_case(f"i1139_small_window_{window // 1000}k", [], steps=steps,
                                 window=window, summary_tokens=2000))
    cases.append(public_case("i1107_first_sight_stub", [], host_overhead=1000,
                             overrides={"large_output_active_replay_stub_threshold_tokens": 4000},
                             steps=[[{"role": "user", "content": "Check records."}, *pairs(12, 500, 8000, "a")],
                                    pairs(1, 100, 20000, "b"), pairs(1, 100, 36000, "c")]))
    cases.append(public_case("i1225_objective_exceeds_room_32k", [
        {"role": "system", "content": "Synthetic system prompt."},
        {"role": "user", "content": "plan" * 22000},
        {"role": "assistant", "content": "Starting work."},
        *pairs(1, 40, 32000)], leaf_history=2))
    return cases + historical_cases()


def historical_cases():
    """Regression inputs whose assembly/provider seams the host loop cannot select."""
    history = [{"role": "system", "content": "Synthetic system prompt."},
               {"role": "user", "content": "Earlier request."},
               {"role": "assistant", "content": "Earlier answer."},
               {"role": "user", "content": "Keep this objective verbatim. " + "plan " * 9600},
               {"role": "assistant", "content": "Starting the requested work."}]
    tail = [row for i in range(10) for row in (
        {"role": "assistant", "content": "Reading.", "tool_calls": [call(f"call-{i}")]},
        {"role": "tool", "content": f"result {i}: " + "data " * 1500,
         "tool_call_id": f"call-{i}", "name": "probe", "tool_name": "probe"},
    )]
    objective = public_case("i1221_historical_assembly_32k", history + tail,
                            leaf_history=4, assembly_tail=20)
    objective["steps"][0][1]["content"] = objective["steps"][0][1]["content"].replace("⟦objective⟧", "")
    objective["steps"][0][3]["content"] = objective["steps"][0][3]["content"].replace("⟦newest⟧", "⟦objective⟧⟦newest⟧")
    # #1225 gives ranges rather than a serialized transcript: 12K at a 16K
    # window with a 4K host system prefix leaves insufficient room even after
    # fitting. The issue specifies ranges, not an exact host transcript.
    oversized = public_case("i1225_historical_assembly_16k", [
        {"role": "system", "content": "rule" * 4000},
        {"role": "user", "content": "plan" * 12000},
        {"role": "assistant", "content": "Starting work."},
        *pairs(1, 40, 32000)], window=16000, leaf_history=2, assembly_tail=2,
        absolute_xfail=True)
    # Same 48-row backlog and 24-row tail as the #1140 regression fixture,
    # followed by the issue's 40 x 9 logical model-call cadence.
    rows = [{"role": "user" if i % 2 == 0 else "assistant",
             "content": (f"FACT-{i} " + "alpha beta gamma delta " * 44) * 2} for i in range(48)]
    rows += [{"role": "user" if i % 2 == 0 else "assistant", "content": f"tail {i}" * 2} for i in range(24)]
    steps = [rows]
    for index in range(1, 360):
        new = pairs(1, 400, 1600, f"historical-{index}-")
        if index % 9 == 0:
            new.insert(0, {"role": "user", "content": f"Continue record {index // 9}."})
        steps.append(new)
    progress = public_case("i1139_historical_sweep_32k", [], steps=steps, window=32000,
                           summary_tokens=2000, fixture_summarizer=True, seed_frontier=4)
    return [objective, oversized, progress]


def generated_cases(n, seed):
    rng = random.Random(seed)
    result = []
    sizes = [1004, 1024, 1044, 1199, 1200, 1201, 2399, 2400, 2401, 3000, 6000, 12000, 48001]
    for i in range(n):
        # At least 70% of samples use small windows, not merely in expectation.
        window = rng.choice([16000, 32000, 64000, 96000, 128000] if i % 10 < 7 else [200000, 1000000])
        size, topology = rng.choice(sizes), rng.choice(["parallel", "sequential", "mixed"])
        # Sample large windows with large rows to bound local row-count cost.
        if window > 128000:
            size = max(size, 12000 if window == 200000 else 48001)
        role = rng.choice(["assistant", "tool", "user"])
        target = int(window * rng.uniform(0.85, 1.5))
        user_chars = (size if role == "user" else 40) + (size if topology == "mixed" else 0)
        count = max(1, math.ceil((window * 0.81 * 4 - user_chars) / (size + 124)),
                    (target * 4 - user_chars) // (size + 124))
        rows = [{"role": "user", "content": body(size) if role == "user" else "Check these records."}]
        if topology == "parallel":
            rows += [{"role": "assistant", "content": body(40), "tool_calls": [call(f"c{j}") for j in range(count)]}]
            rows += [{"role": "tool", "content": body(size), "tool_call_id": f"c{j}", "name": "probe"} for j in range(count)]
        else:
            rows += pairs(count, size if role == "assistant" else 40, 40 if role == "assistant" else size)
            if topology == "mixed":
                rows.insert(len(rows) // 2 // 2 * 2 + 1, {"role": "user", "content": body(size)})
        case = public_case(f"g{i}", rows, family=f"{topology}/{size}/{role}", window=window,
                           config=rng.choice(["defaults", "fleet"]), summarizer=rng.choice(["healthy", "outage"]))
        # Parallel batches amortize call overhead; fill any gap below the density floor.
        with token_counter() as tokens:
            measured = tokens.count_messages_tokens(rows)
        if measured < window * 0.8:
            rows.append({"role": "assistant", "timestamp": float(len(rows) + 1),
                         "content": f"⟦rg{i}.{len(rows)}⟧" + body(math.ceil(window * 0.8 - measured) * 4)})
        result.append(case)
    return result


@contextmanager
def token_counter():
    """Load only the checkout's deterministic counter, without plugin registration."""
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location("differential_counter", root / "__init__.py",
                                                submodule_search_locations=[str(root)])
    if spec.name not in sys.modules:
        sys.modules[spec.name] = importlib.util.module_from_spec(spec)
    from differential_counter import tokens
    with module_patches() as patch:
        patch(tokens, "_encoder_generation", tokens._encoder_generation + 1)
        patch(tokens, "_encoder_ready", True)
        patch(tokens, "_encoder", None)
        yield tokens


def projection_cases(n, seed):
    """Port of p34: preserve its random draw order and uninstrumented limits."""
    rng = random.Random(seed)

    def text(n):
        base = "界面記録確認次手順" if rng.random() < 0.15 else PROSE
        offset = rng.randrange(len(base))
        return (base * (n // len(base) + 2))[offset:offset + n]

    cases = []
    for index in range(n):
        rows, pending = [], []
        for i in range(rng.randint(1, 8)):
            role = "user" if i == 0 and rng.random() < 0.7 else rng.choice(["user", "assistant", "assistant", "tool", "tool"])
            length = rng.choice(LENS) if rng.random() < 0.6 else rng.randint(0, 6000)
            value = text(length)
            row = {"role": role, "timestamp": float(10 + i)}
            if rng.random() < 0.2 and length:
                k = rng.randint(1, 3)
                row["content"] = [{"type": "text", "text": value[j * length // k:(j + 1) * length // k]} for j in range(k)]
                if role == "user" and rng.random() < 0.3:
                    row["content"].append({"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}})
            else:
                row["content"] = value
            if role == "assistant":
                calls = []
                for c in range(rng.choice([0, 0, 1, 1, 2, 3])):
                    cid = f"c{i}_{c}"
                    pending.append(cid)
                    arg = "{}" if rng.random() < 0.6 else json.dumps({"q": text(rng.choice([50, 1100, 1300, 3000]))})
                    calls.append(call(cid))
                    calls[-1]["function"]["arguments"] = arg
                if calls:
                    row["tool_calls"] = calls
            if role == "tool":
                row.update(tool_call_id=pending.pop(0) if pending else f"t{i}", name="probe", tool_name="probe")
            stored = deepcopy(row)
            ext = role in ("assistant", "tool") and rng.random() < 0.1
            if ext:
                stored["content"] = f"[Externalized payload: kind=raw_payload; synthetic placeholder {i}]"
                row["content"] = text(rng.randint(1300, 6000))
            rows.append(dict(message=row, stored=stored, is_stored=rng.random() < 0.9, ext=ext))
        with token_counter() as tokens:
            total = tokens.count_messages_tokens([row["message"] for row in rows])
        limit = rng.choice([0, rng.randint(0, max(total, 1)), rng.randint(0, max(total, 1)), int(total * rng.uniform(0.3, 1.1))])
        for i, entry in enumerate(rows):
            for key in ("message", "stored"):
                if key == "stored" and entry["ext"]:
                    continue
                value = entry[key]["content"]
                tag = f"⟦rp{index}.{i}⟧"
                if isinstance(value, list):
                    value[0]["text"] = tag + value[0]["text"][len(tag):]
                else:
                    entry[key]["content"] = tag + value[len(tag):]
        cases.append(dict(id=f"p{index}", family="p34", layer="projection", rows=rows, limit=limit, window=32768,
                          config="defaults", overrides={"large_output_externalization_threshold_chars": 12000}, summarizer="outage"))
    return cases
