from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


def _load_module(name: str, filename: str):
    path = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


drive_hermes = _load_module("compaction_probe_drive_hermes", "drive_hermes.py")
report_pilot = _load_module("compaction_probe_report_pilot", "report_pilot.py")
score_probes = _load_module("compaction_probe_score_probes", "score_probes.py")


def _jsonl(path: Path, rows: list[object]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_score_probes_covers_three_way_unparseable_and_trap_negative(tmp_path):
    canaries = tmp_path / "canaries.json"
    canaries.write_text(json.dumps({"c-blue": "Blue Key", "c-green": "Green Key"}), encoding="utf-8")
    probes = tmp_path / "probes.jsonl"
    _jsonl(
        probes,
        [
            {"probe_id": "p-correct", "canary_id": "c-blue", "epoch": "E0", "class": "decisions"},
            {"probe_id": "p-abstain", "canary_id": "c-green", "epoch": "E0", "class": "decisions"},
            {"probe_id": "p-hallucinate", "canary_id": "c-green", "epoch": "E1", "class": "paths"},
            {"probe_id": "p-empty", "trap": True, "epoch": "E1", "class": "traps"},
            {"probe_id": "p-trap-abstain", "trap": True, "epoch": "E2", "class": "traps"},
        ],
    )
    results = tmp_path / "results.jsonl"
    _jsonl(
        results,
        [
            {"turn_index": 1, "kind": "probe", "probe_id": "p-correct", "raw_answer": "blue-key"},
            {"turn_index": 2, "kind": "probe", "probe_id": "p-abstain", "raw_answer": "I don't know."},
            {"turn_index": 3, "kind": "probe", "probe_id": "p-hallucinate", "raw_answer": "Red key."},
            {"turn_index": 4, "kind": "probe", "probe_id": "p-empty", "raw_answer": ""},
            {"turn_index": 5, "kind": "probe", "probe_id": "p-trap-abstain", "raw_answer": "not sure"},
        ],
    )

    payload = score_probes.score(results, canaries, probes)

    assert len(payload["probes"]) == 5
    assert [row["classification"] for row in payload["probes"]] == [
        "CORRECT",
        "ABSTAIN",
        "HALLUCINATE",
        "HALLUCINATE",
        "ABSTAIN",
    ]
    assert payload["probes"][3]["unparseable"] is True
    assert payload["probes"][4]["correct_negative"] is True
    assert payload["three_way_totals"] == {"CORRECT": 1, "ABSTAIN": 2, "HALLUCINATE": 2}
    assert payload["totals"]["retention"] == 1 / 3
    assert payload["per_epoch"]["E0"]["total"] == 2


def _write_arm(directory: Path, name: str, classifications: list[str]) -> None:
    directory.mkdir()
    probes = [
        {"probe_id": f"p{index}", "classification": value}
        for index, value in enumerate(classifications, 1)
    ]
    scores = {
        "probes": probes,
        "totals": {
            "total": len(probes),
            "correct": classifications.count("CORRECT"),
            "abstain": classifications.count("ABSTAIN"),
            "hallucinate": classifications.count("HALLUCINATE"),
            "canary_total": len(probes),
            "retention": classifications.count("CORRECT") / len(probes),
        },
        "per_epoch": {"E0": {"correct": classifications.count("CORRECT"), "canary_total": len(probes)}},
        "per_class": {"decisions": {"correct": classifications.count("CORRECT"), "canary_total": len(probes)}},
    }
    (directory / "scores.json").write_text(json.dumps(scores), encoding="utf-8")
    (directory / "report.json").write_text(
        json.dumps({"compaction_count": 2, "stall_total_ms": 99, "total_tokens": 1234}),
        encoding="utf-8",
    )
    (directory / "run.manifest.json").write_text(
        json.dumps({"config_sha256": f"sha-{name}"}), encoding="utf-8"
    )


def test_report_pilot_emits_aa_prime_discordance_section(tmp_path):
    left = tmp_path / "left"
    right = tmp_path / "right"
    _write_arm(left, "R2-A", ["CORRECT", "ABSTAIN"])
    _write_arm(right, "R2-A′", ["CORRECT", "HALLUCINATE"])

    report = report_pilot.render([f"R2-A={left}", f"R2-A′={right}"])

    assert "## A/A′" in report
    assert "discordance_count" in report
    assert "| p2 | ABSTAIN | HALLUCINATE | 1 |" in report
    assert "0.5" in report
    assert "sha-R2-A" in report
    assert "2 | 99 | 1234" in report


def test_drive_hermes_dry_run_probes_only_asserts_config_and_does_not_spawn(tmp_path, capsys):
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    config = hermes_home / "config.yaml"
    config.write_text(
        "context:\n  engine: lcm\nmodel:\n  default: gpt-5.6-sol\n",
        encoding="utf-8",
    )
    probes = tmp_path / "probes.jsonl"
    _jsonl(probes, [{"probe_id": "p1", "prompt": "What?"}])
    raw_log = tmp_path / "raw.pty.log"

    rc = drive_hermes.main(
        [
            "--probes",
            str(probes),
            "--probes-only",
            "--dry-run",
            "--log",
            str(raw_log),
            "--hermes-home",
            str(hermes_home),
            "--expect-engine",
            "lcm",
            "--expect-model",
            "gpt-5.6-sol",
        ]
    )

    output = capsys.readouterr().out
    assert rc == 0
    assert "turn_count: 1" in output
    assert "material_count: 0" in output
    assert "probe_count: 1" in output
    assert "turn_timeout_s: 600" in output
    assert "expected_engine: lcm" in output
    assert "expected_model: gpt-5.6-sol" in output
    assert hashlib.sha256(config.read_bytes()).hexdigest() in output
    assert not raw_log.exists()


def test_driver_loads_canonical_canary_json_shapes_and_rejects_empty(tmp_path):
    shapes = [
        {"c1": "alpha"},
        {"canaries": {"c1": {"value": "alpha"}}},
        [{"canary_id": "c1", "expected": "alpha"}],
    ]
    for index, payload in enumerate(shapes):
        path = tmp_path / f"canaries-{index}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        assert drive_hermes.load_canary_values(path) == ["alpha"]
    empty = tmp_path / "empty.json"
    empty.write_text(json.dumps({"canaries": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="no extractable values"):
        drive_hermes.load_canary_values(empty)


def test_score_rejects_unresolved_positive_canary(tmp_path):
    canaries = tmp_path / "canaries.json"
    canaries.write_text(json.dumps({"known": "value"}), encoding="utf-8")
    probes = tmp_path / "probes.jsonl"
    _jsonl(probes, [{"probe_id": "p-missing", "canary_id": "missing"}])
    results = tmp_path / "results.jsonl"
    _jsonl(
        results,
        [{"kind": "probe", "probe_id": "p-missing", "raw_answer": "I don't know"}],
    )
    with pytest.raises(ValueError, match="positive probe.*unresolved canary"):
        score_probes.score(results, canaries, probes)


def test_score_timeout_is_disclosed_and_excluded_from_retention(tmp_path):
    canaries = tmp_path / "canaries.json"
    canaries.write_text(json.dumps({"c1": "alpha", "c2": "beta"}), encoding="utf-8")
    probes = tmp_path / "probes.jsonl"
    _jsonl(
        probes,
        [
            {"probe_id": "p-timeout", "canary_id": "c1"},
            {"probe_id": "p-correct", "canary_id": "c2"},
            {"probe_id": "p-trap", "trap": True},
        ],
    )
    results = tmp_path / "results.jsonl"
    _jsonl(
        results,
        [
            {
                "kind": "probe",
                "probe_id": "p-timeout",
                "raw_answer": "beta appeared before the deadline",
                "timed_out": True,
            },
            {"kind": "probe", "probe_id": "p-correct", "raw_answer": "beta"},
            {"kind": "probe", "probe_id": "p-trap", "raw_answer": "not sure"},
        ],
    )

    payload = score_probes.score(results, canaries, probes)

    timeout_row = payload["probes"][0]
    assert timeout_row["classification"] == "TIMEOUT"
    assert timeout_row["timed_out"] is True
    assert timeout_row["unparseable"] is True
    assert payload["totals"]["timeout"] == 1
    assert payload["totals"]["canary_total"] == 1
    assert payload["totals"]["retention"] == 1.0


def test_explicit_trap_value_never_scores_correct(tmp_path):
    canaries = tmp_path / "canaries.json"
    canaries.write_text(json.dumps({"c1": "forbidden"}), encoding="utf-8")
    probes = tmp_path / "probes.jsonl"
    _jsonl(probes, [{"probe_id": "p-trap", "trap": True, "canary_id": "c1"}])
    results = tmp_path / "results.jsonl"
    _jsonl(
        results,
        [{"kind": "probe", "probe_id": "p-trap", "raw_answer": "forbidden"}],
    )
    with pytest.warns(UserWarning, match="ignoring it for scoring"):
        payload = score_probes.score(results, canaries, probes)
    assert payload["probes"][0]["classification"] == "HALLUCINATE"
    assert payload["totals"]["correct"] == 0


def test_report_pilot_aborts_for_missing_invalid_or_incomplete_scores(tmp_path):
    arm = tmp_path / "arm"
    arm.mkdir()
    with pytest.raises(ValueError, match="scores artifact"):
        report_pilot.render([f"R2-A={arm}"])
    scores = arm / "scores.json"
    scores.write_text("{", encoding="utf-8")
    with pytest.raises(ValueError, match="scores artifact invalid JSON"):
        report_pilot.render([f"R2-A={arm}"])
    scores.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="missing probes/totals"):
        report_pilot.render([f"R2-A={arm}"])


class _FakeSession:
    instances = []
    boot_ready = True
    answer = ("", True)

    def __init__(self, log_path, *, child_env):
        self.log_path = log_path
        self.child_env = child_env
        self.sent = []
        type(self).instances.append(self)

    def wait_idle(self, _max_wait, _quiet_seconds):
        return type(self).boot_ready

    def send(self, text, _timeout, _quiet_seconds):
        self.sent.append(text)
        return type(self).answer

    def drain_and_close(self, _seconds=30.0):
        return None


def _driver_args(tmp_path, *, material=None, probes=None, **extra):
    home = tmp_path / "hermes"
    home.mkdir(exist_ok=True)
    (home / "config.yaml").write_text(
        "context:\n  engine: lcm\nmodel:\n  default: gpt-5.6-sol\n",
        encoding="utf-8",
    )
    probes = probes or tmp_path / "probes.jsonl"
    if not probes.exists():
        _jsonl(probes, [{"prompt": "probe"}])
    args = [
        "--probes",
        str(probes),
        "--hermes-home",
        str(home),
        "--expect-engine",
        "lcm",
        "--expect-model",
        "gpt-5.6-sol",
        "--log",
        str(tmp_path / "raw.pty.log"),
        "--quiet-seconds",
        "0",
    ]
    if material is not None:
        args.extend(["--material", str(material)])
    for key, value in extra.items():
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                args.append(flag)
        else:
            args.extend([flag, str(value)])
    return args, home


def test_driver_pins_child_home_uses_probe_ordinal_and_aborts_failed_boot(tmp_path, monkeypatch):
    material = tmp_path / "material.jsonl"
    _jsonl(material, [{"prompt": "material"}, {"status": True}])
    probes = tmp_path / "probes.jsonl"
    _jsonl(probes, [{"prompt": "p1"}, {"prompt": "p2"}])
    _FakeSession.instances = []
    _FakeSession.boot_ready = True
    _FakeSession.answer = ("", True)
    monkeypatch.setattr(drive_hermes, "PtySession", _FakeSession)
    args, home = _driver_args(tmp_path, material=material, probes=probes)
    assert drive_hermes.main(args) == 0
    session = _FakeSession.instances[-1]
    assert session.child_env["HERMES_HOME"] == str(home.resolve())
    assert session.sent[:4] == ["material", "/lcm status", "p1", "p2"]
    results = tmp_path / "results.jsonl"
    rows = [json.loads(line) for line in results.read_text(encoding="utf-8").splitlines()]
    assert [row["probe_id"] for row in rows if row["kind"] == "probe"] == [
        "probe-1",
        "probe-2",
    ]

    _FakeSession.instances = []
    _FakeSession.boot_ready = False
    assert drive_hermes.main(args) == drive_hermes.BOOT_FAILURE_EXIT_CODE
    assert _FakeSession.instances[-1].sent == []


def test_driver_config_sha_and_manifest_corpus_identity(tmp_path, capsys):
    material = tmp_path / "material.jsonl"
    _jsonl(material, [{"prompt": "material"}])
    probes = tmp_path / "probes.jsonl"
    _jsonl(probes, [{"prompt": "probe"}])
    canaries = tmp_path / "canaries.json"
    canaries.write_text(json.dumps({"canaries": [{"canary_id": "c1", "value": "alpha"}]}), encoding="utf-8")
    args, home = _driver_args(
        tmp_path,
        material=material,
        probes=probes,
        canaries=canaries,
        dry_run=True,
    )
    config_sha = hashlib.sha256((home / "config.yaml").read_bytes()).hexdigest()
    args.extend(["--expect-config-sha", config_sha, "--manifest", str(tmp_path / "manifest.json")])
    assert drive_hermes.main(args) == 0
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["input_files"]["material"] == {
        "path": str(material),
        "sha256": hashlib.sha256(material.read_bytes()).hexdigest(),
    }
    assert manifest["input_files"]["probes"]["sha256"] == hashlib.sha256(probes.read_bytes()).hexdigest()
    assert manifest["input_files"]["canaries"]["sha256"] == hashlib.sha256(canaries.read_bytes()).hexdigest()
    assert manifest["config_sha256"] == config_sha

    bad_args = [item for item in args if item not in {"--expect-config-sha", config_sha}]
    bad_args.extend(["--expect-config-sha", "0" * 64])
    with pytest.raises(SystemExit) as exc:
        drive_hermes.main(bad_args)
    assert exc.value.code == 2
    assert "config_sha256" in capsys.readouterr().err


def test_tool_detection_requires_hermes_invocation_marker():
    assert drive_hermes.LCM_TOOL_CALL_RE.search("The answer mentions lcm_recall only") is None
    assert drive_hermes.LCM_TOOL_CALL_RE.search("  📞 Tool: lcm_recall({})")
    assert drive_hermes.LCM_TOOL_CALL_RE.search("  ⚡ Concurrent: 2 tool calls — lcm_recall, lcm_grep")


def test_score_probes_gen_material_frozen_schema(tmp_path):
    # Binds the scorer to gen_material.py's actual output schema: canaries as a
    # LIST of {id, class, epoch, value, ...}; probe rows carry only
    # {id, kind, expect, text} — no canary_id (join key = probe id), traps
    # marked via kind: "trap".
    canaries = tmp_path / "canaries.json"
    canaries.write_text(
        json.dumps(
            [
                {"id": "C1-E0-1", "class": "C1", "epoch": "E0", "turn": 1,
                 "char_offset": 61042, "probe": "What did we decide to name the artifact?",
                 "value": "opal-fjord-b20c"},
                {"id": "C2-E1-1", "class": "C2", "epoch": "E1", "turn": 14,
                 "char_offset": 30500, "probe": "What was the build id?",
                 "value": "canyon-7741"},
            ]
        ),
        encoding="utf-8",
    )
    probes = tmp_path / "probes.jsonl"
    _jsonl(
        probes,
        [
            {"id": "C1-E0-1", "kind": "canary", "expect": "value",
             "text": "What did we decide to name the artifact?"},
            {"id": "C2-E1-1", "kind": "canary", "expect": "value",
             "text": "What was the build id?"},
            {"id": "TRAP-04", "kind": "trap", "expect": "ABSTAIN",
             "text": "What was the build id of the failed midnight pipeline?"},
        ],
    )
    results = tmp_path / "results.jsonl"
    _jsonl(
        results,
        [
            {"turn_index": 36, "kind": "probe", "probe_id": "C1-E0-1",
             "raw_answer": "The artifact name is `opal-fjord-b20c`."},
            {"turn_index": 37, "kind": "probe", "probe_id": "C2-E1-1",
             "raw_answer": "It was zephyr-0000."},
            {"turn_index": 38, "kind": "probe", "probe_id": "TRAP-04",
             "raw_answer": "You didn't mention a failed midnight pipeline."},
        ],
    )

    payload = score_probes.score(results, canaries, probes)

    rows = {row["probe_id"]: row for row in payload["probes"]}
    assert rows["C1-E0-1"]["classification"] == "CORRECT"
    assert rows["C1-E0-1"]["canary_id"] == "C1-E0-1"
    assert rows["C1-E0-1"]["epoch"] == "E0"
    assert rows["C1-E0-1"]["class"] == "C1"
    assert rows["C2-E1-1"]["classification"] == "HALLUCINATE"
    assert rows["C2-E1-1"]["epoch"] == "E1"
    assert rows["TRAP-04"]["trap"] is True
    assert rows["TRAP-04"]["classification"] == "ABSTAIN"
    assert payload["totals"]["retention"] == 1 / 2


def test_abstain_patterns_match_real_model_phrasings():
    # These are verbatim abstention answers observed in R2-A / smokes; the
    # original narrow pattern set classified all of them HALLUCINATE.
    real_abstentions = [
        "No build ID was provided for a failed midnight pipeline. The question "
        "is an abstention probe, so the correct answer is: unknown.\n",
        "I don’t have a saved preference for nebula-name prefixes yet.\n",
        "No canonical config location for the shadow-market has been established.\n",
        "We never decided on a name for the glacier-index.\n",
        "You didn't state a canonical config path under `src/` in our earlier messages.\n",
        "It isn't present in the conversation history available to me.\n",
        "I don't know.\n",
    ]
    for answer in real_abstentions:
        classification, _ = score_probes.classify(
            {"raw_answer": answer}, None, True
        )
        assert classification == "ABSTAIN", f"missed abstention: {answer!r}"
    fabricated = "The build id was canyon-7741."
    classification, _ = score_probes.classify({"raw_answer": fabricated}, None, True)
    assert classification == "HALLUCINATE"


def test_metadata_follows_explicit_canary_id_not_probe_id(tmp_path):
    # Legacy schema: probe_id differs from canary_id. epoch/class metadata must
    # resolve through the same key as the VALUE (the explicit canary_id), not
    # the probe's own id.
    canaries = tmp_path / "canaries.json"
    canaries.write_text(
        json.dumps([{"id": "c1", "class": "C3", "epoch": "E1", "value": "amber-key"}]),
        encoding="utf-8",
    )
    probes = tmp_path / "probes.jsonl"
    _jsonl(probes, [{"probe_id": "p1", "canary_id": "c1"}])
    results = tmp_path / "results.jsonl"
    _jsonl(results, [{"turn_index": 1, "kind": "probe", "probe_id": "p1", "raw_answer": "amber-key"}])

    payload = score_probes.score(results, canaries, probes)

    row = payload["probes"][0]
    assert row["classification"] == "CORRECT"
    assert row["epoch"] == "E1"
    assert row["class"] == "C3"


def test_hedged_fabrication_with_known_value_is_hallucinate():
    # RUN-SHEET rule: any concrete answer is HALLUCINATE — an abstention
    # disclaimer followed by a registered canary value must not score ABSTAIN.
    hedged = "You didn't state a build id, but it was probably canyon-7741."
    classification, _ = score_probes.classify(
        {"raw_answer": hedged}, None, True, known_values=["canyon-7741"]
    )
    assert classification == "HALLUCINATE"
    # Same guard on canary probes: hedge + ANOTHER canary's value = substitution.
    classification, _ = score_probes.classify(
        {"raw_answer": hedged}, "opal-fjord", False, known_values=["opal-fjord", "canyon-7741"]
    )
    assert classification == "HALLUCINATE"
    # A pure abstention still scores ABSTAIN with values registered.
    classification, _ = score_probes.classify(
        {"raw_answer": "You didn't state a build id."}, None, True,
        known_values=["canyon-7741"],
    )
    assert classification == "ABSTAIN"
