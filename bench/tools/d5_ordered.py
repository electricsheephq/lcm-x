"""Bit-parallel suffix LCS for END read check 2."""
from __future__ import annotations


def ordered_pairs(row_keys, host_keys):
    """Match bars._ordered with == for exact str, int, float, bytes, None and plain tuples of those values."""
    if not row_keys or not host_keys:
        return []

    matches = dict.fromkeys(row_keys, 0)
    for bit, key in enumerate(reversed(host_keys)):
        # Dict lookup can identify a NaN with itself even though equality does not.
        if key == key and key in matches:
            matches[key] |= 1 << bit

    # The low bits describe growing host suffixes, from right to left. Each bit
    # records a unit increase in LCS length when that host key is included.
    suffix_steps = [0] * (len(row_keys) + 1)
    for i in range(len(row_keys) - 1, -1, -1):
        previous = suffix_steps[i + 1]
        candidates = previous | matches[row_keys[i]]
        # The bit-parallel LCS update selects the first candidate in each
        # segment starting at a shifted previous step (or at the low bit).
        suffix_steps[i] = candidates & ~(candidates - ((previous << 1) | 1))

    pairs = []
    i = j = 0
    host_suffix = (1 << len(host_keys)) - 1
    while i < len(row_keys) and j < len(host_keys):
        shorter_hosts = host_suffix >> 1
        current, below = suffix_steps[i], suffix_steps[i + 1]
        # Follow bars._ordered forward: match first, then skip a stored row on
        # a tie. The popcounts read the same suffix lengths as its DP table.
        if row_keys[i] == host_keys[j] and (current & host_suffix).bit_count() == (
            (below & shorter_hosts).bit_count() + 1
        ):
            pairs.append((i, j))
            i += 1
            j += 1
            host_suffix = shorter_hosts
        elif (below & host_suffix).bit_count() >= (current & shorter_hosts).bit_count():
            i += 1
        else:
            j += 1
            host_suffix = shorter_hosts
    return pairs
