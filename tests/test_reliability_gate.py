"""Collect the focused reliability gate tests in the ordinary tests/ CI job too."""
from bench.instruments.reliability.test_ci_gate import (  # noqa: F401
    test_each_promoted_bar_gates,
    test_existing_gate_and_open_target_behavior_is_unchanged,
    test_native_rotation_and_unsupported_transport_are_not_promoted,
    test_other_continuity_cells_never_gate,
    test_promoted_continuity_bars_are_exact,
    test_promoted_failure_attribution_and_bar_scope,
    test_stamp_only_failures_need_an_open_target,
)
