"""#2856 PR2 B: the token-count cache holds a whole walk within its bounds and every count equals the uncached one.

C1 oracle (cached == ``_count_tokens_core`` under each encoder generation; cache_info/cache_clear as functools'),
C2 bound (200 MB of distinct text: accounted size, entries, tracemalloc growth). Synthetic data only.
"""
import functools
import gc
import random
import threading
import tracemalloc

import hermes_lcm.tokens as tokens


class _Encoder:
    """A stand-in encoder whose count depends on its generation, so a value from the wrong generation shows."""

    def __init__(self, scale):
        self.scale = scale

    def encode(self, text):
        return [0] * (len(text.encode("utf-8", "surrogatepass")) // self.scale + 1)


def _texts(rng, n):
    alphabet = ["a", "b", " ", "é", "検", "\U0001F600", "\ud800", "\udfff"]
    out = []
    for _ in range(n):
        length = rng.choice([0, 1, 7, 1_023, 1_024, 1_025, 4_096, 32_767, 32_768, rng.randrange(1, 2_000)])
        out.append("".join(rng.choice(alphabet) for _ in range(min(length, 64))) * (length // 64 + 1))
    out = [text[:rng.randrange(0, len(text) + 1)] if text else text for text in out]
    return out + rng.sample(out, len(out) // 3)  # duplicates


def test_c1_cached_count_equals_core_across_generations(monkeypatch):
    rng = random.Random(2856)
    texts = _texts(rng, 300)
    tokens._count_tokens_cached.cache_clear()
    for generation, encoder in enumerate([None, _Encoder(3), _Encoder(5), None]):
        monkeypatch.setattr(tokens, "_get_encoder", lambda encoder=encoder: encoder)
        monkeypatch.setattr(tokens, "_encoder_generation", generation)
        for text in rng.sample(texts, len(texts)):
            for _ in range(2):  # miss, then hit
                assert tokens._count_tokens_cached(text, generation) == tokens._count_tokens_core(text)
                if text and len(text) <= tokens._MAX_CACHEABLE_TOKEN_TEXT_CHARS:
                    assert tokens.count_tokens(text) == tokens._count_tokens_core(text)


def test_c1_cache_info_and_clear_match_functools(monkeypatch):
    monkeypatch.setattr(tokens, "_get_encoder", lambda: _Encoder(4))
    reference = functools.lru_cache(maxsize=tokens._TokenCountCache._MAX_ENTRIES)(
        tokens._count_tokens_cached.__wrapped__)
    ours = tokens._count_tokens_cached
    ours.cache_clear()
    rng = random.Random(7)
    texts = _texts(rng, 200)
    for step in range(3):
        for text in texts:
            generation = rng.randrange(3)
            assert ours(text, generation) == reference(text, generation)
            assert ours.cache_info() == reference.cache_info()
        if step == 1:
            ours.cache_clear()
            reference.cache_clear()
            assert ours.cache_info() == reference.cache_info() == functools._CacheInfo(0, 0, 65_536, 0)


def test_c1_threads_never_see_another_key(monkeypatch):
    monkeypatch.setattr(tokens, "_get_encoder", lambda: _Encoder(2))
    tokens._count_tokens_cached.cache_clear()
    texts = _texts(random.Random(11), 120)
    expected = {text: tokens._count_tokens_core(text) for text in texts}
    bad = []

    def worker(seed):
        rng = random.Random(seed)
        for _ in range(2_000):
            text = rng.choice(texts)
            if tokens._count_tokens_cached(text, 0) != expected[text]:
                bad.append(text)
    threads = [threading.Thread(target=worker, args=(seed,)) for seed in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not bad


def test_c2_bound_holds_over_200_mb_of_distinct_text(monkeypatch):
    monkeypatch.setattr(tokens, "_get_encoder", lambda: None)
    cache = tokens._count_tokens_cached
    cache.cache_clear()
    gc.collect()
    tracemalloc.start()
    try:
        before = tracemalloc.get_traced_memory()[0]
        inserted, i, peak_bytes, peak_entries = 0, 0, 0, 0
        while inserted < 200 * 1024 * 1024:
            length = (40, 600, 1_024, 1_025, 9_000, 32_768)[i % 6]
            text = (f"{i:012d}" + "x" * length)[:length]  # distinct, built and dropped here
            cache(text, 0)
            inserted += len(text)
            i += 1
            if i % 512 == 0:
                peak_bytes = max(peak_bytes, cache.accounted_bytes())
                peak_entries = max(peak_entries, cache.cache_info().currsize)
                assert peak_bytes <= 32 * 1024 * 1024 and peak_entries <= 65_536
            del text
        gc.collect()
        growth = tracemalloc.get_traced_memory()[0] - before
    finally:
        tracemalloc.stop()
    assert cache.accounted_bytes() <= 32 * 1024 * 1024
    assert cache.cache_info().currsize <= 65_536
    assert growth <= 40 * 1024 * 1024, growth
    cache.cache_clear()
