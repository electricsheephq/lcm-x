"""I1/I2/I8: long-session moved turns survive once, without losing history or orphaning results."""
import pytest

from tests.test_anchor_unstored_before_cursor import _long_session_class_a_shape


@pytest.mark.parametrize("variant", ["V1", "V2", "V3"])
@pytest.mark.parametrize("own_first", [True, False])
@pytest.mark.parametrize("call_id", ["call_new", "call_é新"])
def test_anchor_long_session_presence_preserves_occurrences(tmp_path, variant, own_first, call_id):
    assert _long_session_class_a_shape(tmp_path, own_first, variant, call_id) == {
        "call": 1, "result": 0 if variant == "V3" else 1, "reply": 1,
        "moved_prompt": 1, "other_added": {}, "lost": {}}
