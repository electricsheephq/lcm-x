"""Permanent I1/I2/I4 origin retention check on the production compress path."""
from tests.test_issue_1592_origin_request import (
    assert_origin, engine as engine, messages,
)


def test_origin_outside_tail_stays_exact(engine):
    result = engine.compress(messages(), force=True)
    assert_origin(result)
    users = engine._durable_real_user_messages()
    assert len(users) == 2
