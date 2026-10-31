"""Mode-aware D8 identities, using synthetic runs without provider calls."""
import json

import pytest

from bench.tools.d8_guard import compare_runs, main
from bench.tools.tests.test_d8_guard import full_rows, write_run
from bench.tools.tests.test_d8_guard_identities import update_report


def pair(tmp_path, **metadata):
    runs = [write_run(tmp_path / name, full_rows()) for name in ("candidate", "baseline")]
    for root in runs:
        update_report(root, lambda report: report.update(metadata))
    return runs


@pytest.mark.parametrize("field,options,allowed", [
    ("questionSetSha256", {}, False),
    ("questionSetSha256", {"subset": True}, True),
    ("questionSetSha256", {"subset": True, "placebo": True}, True),
    ("questionSetSha256", {"reader_treatment": True}, False),
    ("readerPromptVersion", {}, False),
    ("readerPromptVersion", {"placebo": True}, False),
    ("readerPromptVersion", {"subset": True}, False),
    ("readerPromptVersion", {"reader_treatment": True}, True),
    ("readerPromptVersion", {"subset": True, "reader_treatment": True}, True),
])
def test_identity_difference_requires_its_declared_mode(tmp_path, field, options, allowed):
    candidate, baseline = pair(tmp_path, **{field: "original"})
    update_report(candidate, lambda report: report.update({field: "changed"}))
    result = compare_runs(candidate, baseline, **options)
    assert result["INCOMPLETE"] is (not allowed)
    assert result["pass"] is allowed
    assert result["mode"] == ("placebo" if options.get("placebo") else "non-inferiority")
    if allowed:
        assert result["identity_differences"] == {
            field: {"candidate": "changed", "baseline": "original"}}
        assert "reason" not in result
    else:
        assert f"identity mismatch: {field}" in result["reason"]
        assert "identity_differences" not in result


@pytest.mark.parametrize("field", ["questionSetSha256", "readerPromptVersion"])
@pytest.mark.parametrize("options", [{}, {"subset": True}, {"placebo": True},
                                    {"reader_treatment": True},
                                    {"subset": True, "reader_treatment": True}])
@pytest.mark.parametrize("missing", ["candidate", "baseline", "both"])
def test_missing_identity_rules_survive_modes(tmp_path, field, options, missing):
    candidate, baseline = pair(tmp_path)
    if missing != "both":
        present = baseline if missing == "candidate" else candidate
        update_report(present, lambda report: report.update({field: "present"}))
    result = compare_runs(candidate, baseline, **options)
    assert result["INCOMPLETE"] is (missing != "both")
    assert result["pass"] is (missing == "both")
    assert "identity_differences" not in result
    if missing == "both":
        assert field in result["identities_missing_both"]
    else:
        assert f"identity missing: {field}" in result["reason"]


@pytest.mark.parametrize("field,options", [
    ("questionSetSha256", {"subset": True}),
    ("readerPromptVersion", {"reader_treatment": True}),
])
def test_allowed_difference_never_allows_mixed_run(tmp_path, field, options):
    candidate, baseline = pair(tmp_path, **{field: "original"})
    (candidate / "checkpoint.json").write_text(json.dumps({field: "mixed"}))
    result = compare_runs(candidate, baseline, **options)
    assert result["INCOMPLETE"]
    assert not result["pass"]
    assert f"candidate: identity mixed within run: {field}" in result["reason"]
    assert "identity_differences" not in result


def test_disclosures_are_additive_and_only_present_for_differences(tmp_path, capsys):
    candidate, baseline = pair(tmp_path, questionSetSha256="set", readerPromptVersion="reader")
    ordinary = compare_runs(candidate, baseline, subset=True)
    assert compare_runs(candidate, baseline, subset=True, reader_treatment=True) == ordinary
    assert "identity_differences" not in ordinary
    update_report(candidate, lambda report: report.update(
        questionSetSha256="new-set", readerPromptVersion="new-reader"))
    assert main([str(candidate), str(baseline), "--subset", "--reader-treatment"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["pass"]
    assert result["identity_differences"] == {
        "questionSetSha256": {"candidate": "new-set", "baseline": "set"},
        "readerPromptVersion": {"candidate": "new-reader", "baseline": "reader"}}
    assert {key: value for key, value in result.items() if key != "identity_differences"} == ordinary


def test_reader_treatment_and_placebo_are_usage_error(capsys):
    with pytest.raises(SystemExit) as error:
        main(["absent-candidate", "absent-baseline", "--reader-treatment", "--placebo"])
    assert error.value.code == 2
    output = capsys.readouterr()
    assert not output.out
    assert "not allowed with argument" in output.err
    with pytest.raises(ValueError, match="reader-treatment.*placebo"):
        compare_runs("absent-candidate", "absent-baseline", reader_treatment=True, placebo=True)
