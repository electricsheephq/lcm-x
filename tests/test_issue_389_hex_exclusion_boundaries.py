"""#389: pin the edges of the pure-hex exclusion in the private-key proximity backstop.

`_pem_fragment_near_private_key_placeholder` blocks any 16+ base64-alphabet run near a
private_key placeholder, except a run that is entirely hex of 32 to 128 characters
(a git SHA or a hash digest). The dispatch tests in test_embedding_privacy.py only
check that a 40-character SHA still embeds. These tests pin the edges, so that
widening the hex range or matching hex anywhere inside a run fails a test.
"""

import pytest

from hermes_lcm.ingest_protection import _pem_fragment_near_private_key_placeholder

PLACEHOLDER = "[LCM embedding privacy: name=private_key]"
HEX = "0123456789abcdef" * 9  # 144 hex characters to slice from


def _near_placeholder(run):
    return f"context before\n{PLACEHOLDER}\nDeployed at commit {run} per the runbook."


@pytest.mark.parametrize("length", [32, 40, 64, 128])
def test_a_pure_hex_run_of_32_to_128_characters_does_not_block(length):
    assert (
        _pem_fragment_near_private_key_placeholder(_near_placeholder(HEX[:length]))
        is False
    )


@pytest.mark.parametrize("length", [16, 31, 129])
def test_a_pure_hex_run_outside_32_to_128_characters_still_blocks(length):
    assert (
        _pem_fragment_near_private_key_placeholder(_near_placeholder(HEX[:length]))
        is True
    )


def test_a_mixed_run_that_contains_a_40_character_hex_substring_still_blocks():
    run = "Zq" + HEX[:40]
    assert _pem_fragment_near_private_key_placeholder(_near_placeholder(run)) is True


def test_a_hex_run_far_from_the_placeholder_is_out_of_scope():
    # Control: the run is outside the 160-character window, so nothing blocks either way.
    text = f"{PLACEHOLDER}{' ' * 400}{HEX[:31]}"
    assert _pem_fragment_near_private_key_placeholder(text) is False
