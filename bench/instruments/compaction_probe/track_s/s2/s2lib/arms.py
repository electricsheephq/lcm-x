"""Arm registry: every arm is a set of the engine's own `LCM_*` env inputs (config.py ENV_FIELD_SPECS), never a patch."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path

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
# D5-B (2026-10-12): fleet arm with a 5,000-token leaf chunk (v0.26-sized summary calls with whole messages, #611).
BASE_ARMS["LCMX-fleet-chunk5k"] = ({"LCM_LEAF_CHUNK_TOKENS": "5000"}, "fleet + leaf chunk 5000 tokens (D5-B)")
# P0 chunk cell (2026-10-12): 120 s timeout + leaf summary target capped at 2000 tokens; only the leaf chunk differs.
for _n, _c in (("LCMX-fleet-120s-c8k", "8000"), ("LCMX-fleet-120s-c16k", "16000"), ("LCMX-fleet-120s-c24k", "24000")):
    BASE_ARMS[_n] = ({"LCM_SUMMARY_TIMEOUT_MS": "120000", "LCM_LEAF_CHUNK_TOKENS": _c, "LCM_LEAF_TARGET_MAX_TOKENS": "2000",
                     "LCM_SUMMARY_REASONING_EFFORT": "low"},
                     f"fleet + 120 s + leaf chunk {_c} + leaf target max 2000 , effort low (P0 chunk cell)")
# rc3 per-message-wait gate (#1128): the fleet policy with the leaf chunk and the summary timeout left to the product
# defaults (v0.27.0-rc3: chunk 24000, leaf target max 2000, timeout 120 s). A None value removes the key from the env.
BASE_ARMS["LCMX-fleet-rc3"] = ({"LCM_LEAF_CHUNK_TOKENS": None, "LCM_SUMMARY_TIMEOUT_MS": None,
                                "LCM_SUMMARY_REASONING_EFFORT": "low"},
                               "fleet policy, product leaf chunk / target cap / timeout defaults, effort low (#1128)")
# D8 prompt lane (2026-10-05): the shipped fleet arm with the opt-in v2 summariser prompts (#646).
BASE_ARMS["LCMX-fleet-v2"] = ({"LCM_SUMMARY_PROMPT_VERSION": "2"}, "fleet + summariser prompt v2 (#646)")
CONTROLS = {"C0": "no compaction: the full admitted transcript as context; UNAVAILABLE when it does not fit the reader",
            "C1": "tail only: the last LCM_FRESH_TAIL_COUNT (24) rows of the stream, no summaries, no store"}
# Eval-2 arms run the fleet's actual stub threshold (10000, the product default); FLEET keeps 6000 for v3 comparability.
BASE_ARMS.update({n: ({"LCM_LARGE_OUTPUT_ACTIVE_REPLAY_STUB_THRESHOLD_TOKENS": "10000"},
                      "Eval-2 pinned product arm; ablations live in the pin") for n in ("L0", "L1", "L1-H", "L1-noptr")})

# D5-A (2026-10-14): the reference agent's own `LCM_*` set, read from the file named by S2_ARM_ENV_FILE (KEY=VALUE lines).
ENV_FILE_ARM = "LCMX-env-file"
HARNESS_KEYS = ("LCM_DATABASE_PATH", "LCM_HERMES_BASE_DIR")  # set inside the run dir by the harness; file values ignored
SECRET_ENV: dict = {}  # VOYAGE_API_KEY from the file (embeddings on only); applied to the process env, never recorded

def env_file_arm() -> dict:
    """Only `LCM_*` keys are taken, plus VOYAGE_API_KEY when the file turns embeddings on; every other key is ignored and
    named in env_file.ignored_keys. No fleet value is merged in."""
    path = os.environ.get("S2_ARM_ENV_FILE")
    if not path:
        raise SystemExit(f"{ENV_FILE_ARM} needs S2_ARM_ENV_FILE")
    path = Path(path).resolve()
    raw = dict(ln.split("=", 1) for ln in path.read_text().splitlines()
               if ln.strip() and not ln.lstrip().startswith("#") and "=" in ln)
    raw = {k.strip(): v.strip() for k, v in raw.items()}
    on = raw.get("LCM_EMBEDDINGS_ENABLED", "").lower() in ("1", "true", "yes", "on")
    env = {k: v for k, v in raw.items() if k.startswith("LCM_") and k not in HARNESS_KEYS}
    SECRET_ENV.clear()
    if on and raw.get("VOYAGE_API_KEY"):
        SECRET_ENV["VOYAGE_API_KEY"] = raw["VOYAGE_API_KEY"]
    return {"name": ENV_FILE_ARM, "base": ENV_FILE_ARM, "open": False, "kind": "plain", "env": env,
            "note": "reference agent LCM_* set from S2_ARM_ENV_FILE (D5-A)", "unsupported": None,
            "env_file": {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "keys": len(env),
                         "ignored_keys": sorted(k for k in raw if k not in env and k not in SECRET_ENV),
                         "voyage_key_from_file": bool(SECRET_ENV)}}

def all_arms() -> list[str]:
    return [a for b in BASE_ARMS for a in (b, b + "-open")] + list(CONTROLS)

def resolve(arm: str) -> dict:
    """{name, base, open, kind, env|None, note, unsupported|None}"""
    if arm == ENV_FILE_ARM:
        return env_file_arm()
    pins = json.loads(os.environ.get("S2_ARM_PINS", "{}"))  # {arm: {worktree, sha}}; supplied by the orchestrator
    if arm.removesuffix("-open") in ("L0", "L1", "L1-H", "L1-noptr", "H") and not all(
            pins.get(arm.removesuffix("-open"), {}).get(k) for k in ("worktree", "sha")):
        raise SystemExit(f"Eval-2 requires S2_ARM_PINS entry for {arm}")
    if arm == "H":
        return dict(name=arm, base=arm, open=False, kind="builtin", env={}, note="Hermes default lean", unsupported=None,
                    **pins.get(arm, {}))
    if arm in CONTROLS:
        return {"name": arm, "base": arm, "open": False, "kind": "control", "env": dict(FLEET), "note": CONTROLS[arm],
                "unsupported": None}
    base, is_open = (arm[:-5], True) if arm.endswith("-open") else (arm, False)
    if base not in BASE_ARMS:
        raise SystemExit(f"unknown arm {arm}; known: {', '.join(all_arms())}")
    delta, note = BASE_ARMS[base]
    return {"name": arm, "base": base, "open": is_open, "kind": "open" if is_open else "plain",
            "env": None if delta is None else {k: v for k, v in {**FLEET, **delta}.items() if v is not None},
            "note": note, "unsupported": UNSUPPORTED.get(base),
            **pins.get(base, {})}
