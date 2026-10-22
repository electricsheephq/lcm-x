"""Golden JSON bytes generated from the candidate's actual emitted views."""

import json

import pytest

from tests.compat.helpers import CASES, GOLDEN, emit_case, open_engine


@pytest.mark.parametrize("case", CASES)
def test_projection_bytes(tmp_path, case):
    engine = open_engine(tmp_path)
    try:
        emitted, _ = emit_case(engine, case)
        actual = json.dumps(emitted, sort_keys=True)
        expected = json.loads(GOLDEN.read_text())[case]
        assert actual.encode("utf-8") == expected.encode("utf-8")
    finally:
        engine.shutdown()
