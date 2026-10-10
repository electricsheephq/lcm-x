"""Arm registry: every arm is a set of the engine's own `LCM_*` env inputs (config.py ENV_FIELD_SPECS), never a patch."""
from __future__ import annotations

# The managed fleet policy (PCS scripts/rollout-lcm-x-profile.sh `desired`, lines 198-205 of pcs-golden-build-e528dd53),
# minus temporal rollups (FLEET_EXCLUDED; labelled). LCM_NATIVE_RECOVERY=false matches the fleet default (S8). Timeout 60 s = the engine default, set explicitly so config.json shows it.
FLEET = {
    "LCM_CONTEXT_THRESHOLD": "0.75", "LCM_FRESH_TAIL_COUNT": "24", "LCM_FRESH_TAIL_MAX_TOKENS": "24000",
    "LCM_LEAF_CHUNK_TOKENS": "8000", "LCM_THRESHOLD_FULL_SWEEP_ENABLED": "true",
    "LCM_LARGE_OUTPUT_EXTERNALIZATION_ENABLED": "true", "LCM_LARGE_OUTPUT_EXTERNALIZATION_THRESHOLD_CHARS": "12000",
    "LCM_LARGE_OUTPUT_ACTIVE_REPLAY_STUBBING_ENABLED": "true",
    "LCM_LARGE_OUTPUT_ACTIVE_REPLAY_STUB_THRESHOLD_TOKENS": "6000",
    "LCM_SUMMARY_TIMEOUT_MS": "60000", "LCM_NATIVE_RECOVERY": "false",  # S8: fleet default false (PCS 8a98585)
    "LCM_SUMMARY_SPEND_MAX_CALLS": "0",  # S8 HARNESS OVERRIDE, see HARNESS_OVERRIDES
    "LCM_TEMPORAL_ROLLUPS_ENABLED": "false",  # FLEET_EXCLUDED, explicit: the product default is on (#1013)
}
HARNESS_OVERRIDES = {"LCM_SUMMARY_SPEND_MAX_CALLS=0": "spend guard off: the replay feeds 305 rows in ~9 min, so the "
                     "24 calls / 600 s guard trips from replay speed alone (field: 0 trips on 22 profiles); per-run "
                     "600 s call counts are reported beside it"}
FLEET_EXCLUDED = {
    "LCM_TEMPORAL_ROLLUPS_ENABLED=true": "background rollup maintenance, not compaction; left off so compress() "
                                        "timing and summariser-call counts contain compaction work only",
}

UNSUPPORTED = {
    "LCMX-ref-v2": "LCM_SUMMARY_PROMPT_VERSION does not exist at 9edfa46 (no prompt_version in config.py or escalation.py)",
    "LCMX-fleet-clipoff": "the 2,000 + 800-char summariser input clip is hard-coded in LCMEngine._serialize_messages "
                          "(engine.py:6326-6327 at 9edfa46); no config input disables it",
}

BASE_ARMS = {
    "LCMX-fleet": ({}, "managed fleet policy"),
    "LCMX-fleet-agedoff": ({"LCM_LARGE_OUTPUT_ACTIVE_REPLAY_STUB_AGED_THRESHOLD_TOKENS": "6000"},
                           "fleet + aged threshold equal to first-sight 6000 (rc3 engine.py:6794-6801)"),
    "LCMX-fleet-120s": ({"LCM_SUMMARY_TIMEOUT_MS": "120000"}, "fleet + 120 s summariser timeout"),
    "LCMX-ref": ({"LCM_SUMMARY_TIMEOUT_MS": "120000", "LCM_SUMMARY_REASONING_EFFORT": "medium",
                  "LCM_RESERVE_TOKENS_FLOOR": "40000"}, "reference profile: 8k, 120 s, effort medium, floor 40k"),
    "LCMX-ref-v2": (None, "UNSUPPORTED"),
    "LCMX-fleet-clipoff": (None, "UNSUPPORTED"),
    "LCMX-fleet-low": ({"LCM_SUMMARY_REASONING_EFFORT": "low"}, "fleet + summary effort low (the one speed arm)"),
    "LCMX-fleet-stub10k": ({"LCM_LARGE_OUTPUT_ACTIVE_REPLAY_STUB_THRESHOLD_TOKENS": "10000"},
                           "fleet + stub threshold 10k (Codex per-output cap; the new first-sight default)"),
    "LCMX-fleet-stub2k": ({"LCM_LARGE_OUTPUT_ACTIVE_REPLAY_STUB_THRESHOLD_TOKENS": "2000"},
                          "UPPER BOUND on the planned aged tier (needs v0.24.9): EVERY stub is at 2k here, not only aged"),
}
# D8 prompt lane (2026-10-05): the shipped fleet arm with the opt-in v2 summariser prompts (#646).
BASE_ARMS["LCMX-fleet-v2"] = ({"LCM_SUMMARY_PROMPT_VERSION": "2"}, "fleet + summariser prompt v2 (#646)")
CONTROLS = {"C0": "no compaction: the full admitted transcript as context; UNAVAILABLE when it does not fit the reader",
            "C1": "tail only: the last LCM_FRESH_TAIL_COUNT (24) rows of the stream, no summaries, no store"}

def all_arms() -> list[str]:
    return [a for b in BASE_ARMS for a in (b, b + "-open")] + list(CONTROLS)

def resolve(arm: str) -> dict:
    """{name, base, open, kind, env|None, note, unsupported|None}"""
    if arm in CONTROLS:
        return {"name": arm, "base": arm, "open": False, "kind": "control", "env": dict(FLEET), "note": CONTROLS[arm],
                "unsupported": None}
    base, is_open = (arm[:-5], True) if arm.endswith("-open") else (arm, False)
    if base not in BASE_ARMS:
        raise SystemExit(f"unknown arm {arm}; known: {', '.join(all_arms())}")
    delta, note = BASE_ARMS[base]
    return {"name": arm, "base": base, "open": is_open, "kind": "open" if is_open else "plain",
            "env": None if delta is None else {**FLEET, **delta}, "note": note, "unsupported": UNSUPPORTED.get(base)}
