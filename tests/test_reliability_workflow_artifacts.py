"""Keep hosted reliability cells usable by the standalone re-scorer."""

from pathlib import Path
import re
import shlex


WORKFLOW = Path(__file__).resolve().parents[1] / ".github/workflows/reliability-nightly.yml"


def test_every_matrix_run_keeps_nonpassing_cell_databases():
    workflow = WORKFLOW.read_text(encoding="utf-8")
    commands = [line for line in workflow.splitlines() if "run_matrix.py" in line]
    assert len(commands) == 3
    for command in commands:
        args = shlex.split(command)
        assert "--keep-dbs" in args
        assert args[args.index("--keep-dbs") + 1] == "fail"
        assert "--keep-homes" not in args


def test_upload_keeps_whole_cells_without_caches_or_tool_files():
    workflow = WORKFLOW.read_text(encoding="utf-8")
    upload = workflow.split("      - name: Upload results\n", 1)[1]
    paths_block = upload.split("          path: |\n", 1)[1].split("          if-no-files-found:", 1)[0]
    paths = {line.strip() for line in paths_block.splitlines() if line.strip()}
    assert "reliability-out/*/cells/**" in paths
    assert "reliability-out/default-window-*/**" in paths
    assert "!reliability-out/**/pycache/**" in paths
    assert "!reliability-out/*/cells/*/*/*/files/**" in paths
    # A file allowlist silently drops future scorer inputs. Payload-only Hermes
    # home wrappers must remain: release_scratch keeps externalized rows there.
    assert not [path for path in paths if re.search(r"/cells/.*\.[a-z]+$", path)]
    assert not any("hermes-home" in path for path in paths)


def test_default_window_has_no_report_only_switch():
    workflow = WORKFLOW.read_text(encoding="utf-8")
    assert "DEFAULT_WINDOW_REPORT_ONLY" not in workflow


def test_default_window_command_failure_is_not_ignored():
    workflow = WORKFLOW.read_text(encoding="utf-8")
    step = re.search(r"(?m)^      - name: Default-window[^\n]*\n((?:^[^\S\n]{8}.*\n)+)", workflow)
    assert step and "continue-on-error" not in step.group()


def job(workflow, name):
    match = re.search(rf"(?ms)^  {name}:\n(.*?)(?=^  \w+:|\Z)", workflow)
    assert match, f"{name} job missing"
    return match.group(1)


def test_dispatch_accepts_a_named_baseline_and_stages_tables():
    workflow = WORKFLOW.read_text(encoding="utf-8")
    dispatch = re.search(r"(?ms)^  workflow_dispatch:\n(.*?)(?=^  \w+:)", workflow).group(1)
    assert re.search(r"(?ms)^      baseline_run:\n.*?^        type: string\n.*?^        required: false$", dispatch)
    compare = job(workflow, "compare")
    condition = "github.event_name == 'schedule' || github.event_name == 'workflow_dispatch'"
    assert condition in re.search(r"(?m)^    if: .+$", compare).group()
    staging = re.search(r"(?m)^      - name: Stage tables.*\n        if: .+$", job(workflow, "matrix")).group()
    assert condition in staging
    assert "run: python -m bench.instruments.reliability.ci alarm --read" in compare
    assert re.search(r"(?ms)name: Upload dispatch tables.*?if: .*?workflow_dispatch.*?name: reliability-tables\n", compare)


def test_workflow_concurrency_is_separate_by_event_and_only_prs_cancel():
    workflow = WORKFLOW.read_text(encoding="utf-8")
    concurrency = re.search(r"(?ms)^concurrency:\n(.*?)(?=^\w+:)", workflow).group(1)
    assert "  group: reliability-${{ github.ref }}-${{ github.event_name }}\n" in concurrency
    assert "  cancel-in-progress: ${{ github.event_name == 'pull_request' }}\n" in concurrency


def test_alarm_is_schedule_only_and_uses_the_compare_python():
    workflow = WORKFLOW.read_text(encoding="utf-8")
    alarm = job(workflow, "alarm")
    assert re.search(r"(?m)^    if: (.+)$", alarm).group(1) == (
        "${{ !cancelled() && github.event_name == 'schedule' && needs.compare.outputs.alarm != '' }}"
    )
    python = re.search(r"(?m)^      - uses: actions/setup-python.*\n        with:\n          python-version: .+$",
                       job(workflow, "compare")).group()
    assert python in alarm


def test_scheduled_tables_publish_only_after_judgment_or_bootstrap():
    workflow = WORKFLOW.read_text(encoding="utf-8")
    publish = job(workflow, "publish")
    assert "needs: [compare, alarm]" in publish
    condition = re.search(r"(?ms)^    if: >-\n(.*?)^    runs-on:", publish).group(1)
    assert " ".join(condition.split()) == (
        "${{ !cancelled() && github.event_name == 'schedule' && "
        "(needs.compare.outputs.bootstrap == 'success' || needs.compare.outputs.status == '0' || "
        "(needs.compare.outputs.status == '1' && needs.alarm.result == 'success')) }}"
    )
    compare = job(workflow, "compare")
    assert re.search(r"(?ms)name: Validate bootstrap tables.*?^        id: bootstrap\n", compare)
    assert "bootstrap: ${{ steps.bootstrap.outcome }}" in compare
    assert "name: reliability-tables-pending" in compare and "name: reliability-tables-pending" in publish
    assert "name: reliability-tables\n" in publish
