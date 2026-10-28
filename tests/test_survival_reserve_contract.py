"""Pin reserve coercion, clamping and rounding through both real engine callers."""

from types import SimpleNamespace

import pytest

import hermes_lcm.survival_fit as survival_fit
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine


WINDOWS = (0, 1, 16_384, 32_768, 128_000)
# Literal outputs preserve float rounding, especially the 128K ceiling at 90% reserve.
NO_RESERVE = (None, 1, 16_384, 32_768, 128_000)
DEFAULT_RESERVE = (None, 0, 13_926, 27_852, 108_800)
HALF_RESERVE = (None, 0, 8_192, 16_384, 64_000)
MAX_RESERVE = (None, 0, 1_638, 3_276, 12_799)
STRING_RESERVE = (None, 0, 13_107, 26_214, 102_400)
MISSING = object()


@pytest.fixture
def engine(tmp_path, monkeypatch):
    # Empty messages have no host overhead beyond the explicit observed count.
    monkeypatch.setattr(survival_fit, "_host_estimate", lambda messages: 0)
    instance = LCMEngine(config=LCMConfig(database_path=str(tmp_path / "lcm.db")))
    try:
        yield instance
    finally:
        instance.shutdown()


@pytest.mark.parametrize("window_cap", [None, 16_384])
@pytest.mark.parametrize("request_cap", [None, 4_000])
@pytest.mark.parametrize("window_index", range(len(WINDOWS)), ids=[str(w) for w in WINDOWS])
@pytest.mark.parametrize(("reserve", "ceilings"), [
    pytest.param(MISSING, DEFAULT_RESERVE, id="missing"),
    pytest.param(None, NO_RESERVE, id="none"),
    pytest.param(0, NO_RESERVE, id="integer-zero"),
    pytest.param(0.0, NO_RESERVE, id="float-zero"),
    pytest.param(-0.5, NO_RESERVE, id="negative"),
    pytest.param(0.15, DEFAULT_RESERVE, id="default"),
    pytest.param(0.5, HALF_RESERVE, id="half"),
    pytest.param(0.9, MAX_RESERVE, id="maximum"),
    pytest.param(0.95, MAX_RESERVE, id="above-maximum"),
    pytest.param(5, MAX_RESERVE, id="large"),
    pytest.param("0.2", STRING_RESERVE, id="numeric-string"),
    pytest.param("abc", None, id="invalid-string"),
    pytest.param(float("nan"), NO_RESERVE, id="nan"),
    pytest.param(float("inf"), MAX_RESERVE, id="infinity"),
])
def test_survival_reserve_callers(engine, monkeypatch, reserve, ceilings, window_index,
                                 window_cap, request_cap):
    config_values = vars(engine._config).copy()
    config_values.pop("survival_reserve")
    if reserve is not MISSING:
        config_values["survival_reserve"] = reserve
    # Deleting a dataclass instance attribute would still expose its class default.
    monkeypatch.setattr(engine, "_config", SimpleNamespace(**config_values))
    engine.context_length = window = WINDOWS[window_index]

    if window == 0:
        assert engine._survival_ceiling() is None
        assert engine._survival_fit_budget([], 17, window_cap, request_cap) is None
    elif ceilings is None:
        for call in (
            engine._survival_ceiling,
            lambda: engine._survival_fit_budget([], 17, window_cap, request_cap),
        ):
            with pytest.raises(ValueError) as error:
                call()
            assert str(error.value) == "could not convert string to float: 'abc'"
    else:
        assert engine._survival_ceiling() == ceilings[window_index]
        budget_index = min(window_index, 2) if window_cap is not None else window_index
        ceiling = ceilings[budget_index]
        if request_cap is not None:
            ceiling = min(ceiling, request_cap)
        overhead = min(WINDOWS[budget_index] // 2, 17)
        assert engine._survival_fit_budget([], 17, window_cap, request_cap) == max(1, ceiling - overhead)
