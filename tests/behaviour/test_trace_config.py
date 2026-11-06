"""Config trace fingerprints preserve overrides and default changes (#1894)."""
import dataclasses
from pathlib import Path

import pytest

from hermes_lcm.config import LCMConfig
from tests.behaviour.assembly_trace_harness import Canonical as AssemblyCanonical
from tests.behaviour.compress_trace_harness import Canonical as CompressCanonical
from tests.behaviour.ingest_trace_harness import IngestCanonical
from tests.behaviour.reconcile_trace_harness import Canonical as ReconcileCanonical


@pytest.fixture(params=[AssemblyCanonical, CompressCanonical, IngestCanonical, ReconcileCanonical])
def canonical(request, tmp_path):
    cls = request.param
    return cls(tmp_path, []) if cls in {IngestCanonical, ReconcileCanonical} else cls(tmp_path)


def test_new_default_off_field_keeps_fingerprint(canonical, monkeypatch):
    before = canonical.value(LCMConfig())
    field = dataclasses.fields(dataclasses.make_dataclass("Future", [("future_off", bool, False)]))[0]
    monkeypatch.setitem(LCMConfig.__dataclass_fields__, field.name, field)
    monkeypatch.setattr(LCMConfig, field.name, False, raising=False)
    assert canonical.value(LCMConfig()) == before
    enabled = LCMConfig()
    enabled.future_off = True
    assert canonical.value(enabled) == {**before, "future_off": True}


def test_changed_existing_default_moves_fingerprint(canonical, monkeypatch):
    before = canonical.value(LCMConfig())
    original = LCMConfig.__init__

    def changed_default(self, *args, **kwargs):
        kwargs.setdefault("fresh_tail_count", 99)
        original(self, *args, **kwargs)

    monkeypatch.setattr(LCMConfig, "__init__", changed_default)
    assert canonical.value(LCMConfig()) == {**before, "fresh_tail_count": 99}


def test_scenario_override_is_visible(canonical):
    assert canonical.value(LCMConfig(fresh_tail_count=2)) == {
        "__trace_config__": 1, "fresh_tail_count": 2,
    }


def test_other_dataclasses_keep_full_serialization(canonical):
    record = dataclasses.make_dataclass("Record", [("enabled", bool, False), ("paths", list, dataclasses.field(default_factory=list))])
    assert canonical.value(record(paths=[Path("relative")])) == {"enabled": False, "paths": ["relative"]}


def test_baseline_lists_every_config_field():
    """A new LCMConfig field must add its default to the pinned baseline in the same PR.

    That changes no golden (a field at its baseline default is omitted), and a later change to
    that default then moves the fingerprint instead of tracking the live default silently.
    """
    from tests.behaviour.trace_config import BASELINE
    fields = {field.name for field in dataclasses.fields(LCMConfig)}
    missing, stale = sorted(fields - set(BASELINE["defaults"])), sorted(set(BASELINE["defaults"]) - fields)
    assert not missing and not stale, (
        f"update tests/fixtures/trace_config_baseline.json: add {missing} at their defaults, remove {stale}")
