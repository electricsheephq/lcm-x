"""Keep optional installed result-writer checks explicit on clean runners."""
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from bench.fmp import closeout, derived, dev
from benchmarking.standalone import ensure_agent_context_engine_importable

ensure_agent_context_engine_importable()

CARD_INTEGRATION = {
    "test_packaged_card_contains_only_identity_specs",
    "test_score_emits_card_and_preserves_rescore_inputs",
    "test_10_mock_e2e_snapshots_warm_resume_and_corruption",
    "test_12_cli_score_activation_missing_baseline_and_deadline",
    "test_1_deadline_settles_and_scores",
    "test_7_card_uses_each_scored_denominator",
}


class Card:
    """Same no-op writer contract used by the existing synthetic dev fixtures."""
    def parser(self):
        return SimpleNamespace(parse_args=lambda args: args)

    def execute(self, args):
        return 0


@pytest.fixture(autouse=True)
def installed_result_card(monkeypatch, request):
    original = closeout.card_tool

    def available():
        path = os.environ.get("FMP_RESULT_CARD_TOOL")
        if not path or not Path(path).is_file():
            if request.node.originalname in CARD_INTEGRATION:
                pytest.skip("FMP_RESULT_CARD_TOOL must point to an installed bench-card.py")
            return Card()
        return original()

    for module in (closeout, derived, dev):
        monkeypatch.setattr(module, "card_tool", available)
