"""Keep synthetic driver assertions running without the optional installed card tool."""
import os
from pathlib import Path
import sys

import pytest


@pytest.fixture(autouse=True)
def installed_result_card(monkeypatch, request):
    path = os.environ.get("FMP_RESULT_CARD_TOOL")
    if path and Path(path).is_file():
        return
    if request.node.originalname == "test_completed_fake_run_has_validated_result_card":
        pytest.skip("FMP_RESULT_CARD_TOOL must point to an installed bench-card.py")
    import test_d2_calibrated
    original = test_d2_calibrated.load

    def load(name):
        module = original(name)
        if name == "d2_driver":
            module.result_card = lambda *args: None
        return module

    # Existing tests import load by value; patch those aliases without editing them.
    for module in tuple(sys.modules.values()):
        if getattr(module, "__file__", None) and Path(module.__file__).parent == Path(__file__).parent:
            if getattr(module, "load", None) is original:
                monkeypatch.setattr(module, "load", load)
