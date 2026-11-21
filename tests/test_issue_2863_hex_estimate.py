"""#2863 (E''): without a tokenizer, long hex payloads are sized at 2 chars/token; everything else is unchanged.

Expected values are the issue's probe values (probe v3, `cand_estimate.make_hex65(2.0)`) on the same synthetic
samples. Only the reporter-shaped hex trace rows move; digests, SHAs, LCM's own pointer names, base64 and every
non-ASCII sample keep the previous estimate."""

import base64
import json
import random
import uuid

import pytest

from hermes_lcm import tokens as T
from hermes_lcm.tokens import _fallback_token_estimate

pytestmark = pytest.mark.issue(2863)
N = 20_000


def legacy_estimate(text):
    """The estimate before E'' (tokens.py at 9452166)."""
    length = len(text)
    if length == 0:
        return 0
    if text.isascii():
        return length // 4 + 1
    ratio = sum(1 for ch in text if ord(ch) > 127) / length
    return int(length / (1.5 if ratio >= 0.5 else 2.5 if ratio >= 0.2 else 4)) + 1


def fit(s, n=N):
    return (s * (n // max(1, len(s)) + 1))[:n]


def trace_rows(seed, chars):
    rnd = random.Random(seed)
    rows, n, total = [], 0, 0
    while total < chars:
        row = (f"trace {seed % 100:02d}.{n:04d} op=frame ts=17000{n:05d} payload="
               + "".join(rnd.choice("0123456789abcdef") for _ in range(96)))
        rows.append(row)
        total += len(row) + 1
        n += 1
    return "\n".join(rows)[:chars]


def samples():
    rnd = random.Random(2863)
    words = ("the parser reads each line of input and checks that timestamps increase while a missing field "
             "is recorded as an error before the loop moves on to the next record in the batch").split()
    prose = " ".join(rnd.choice(words) for _ in range(5000)).capitalize()
    code = "".join(
        f"def parse_{i}(line, sep=','):\n    fields = line.split(sep)\n    if len(fields) < {i % 7 + 2}:\n"
        f"        raise ValueError(f'bad line {{line!r}}')\n    return Row{i}(*fields)\n\n" for i in range(400))
    js = json.dumps([{"id": str(uuid.UUID(int=rnd.getrandbits(128), version=4)), "name": f"record-{i}",
                      "status": rnd.choice(["ok", "failed", "pending"]), "ts": 1700000000 + i} for i in range(400)],
                    indent=1)
    hexrows = trace_rows(7, N)
    b64 = base64.b64encode(rnd.randbytes(N)).decode()
    git = "\n".join(f"commit {rnd.getrandbits(160):040x}\nAuthor: dev <dev@example.invalid>\n\n    fix parser edge case {i}\n"
                    for i in range(400))
    zh = "解析器读取输入文件的每一行，将其拆分为字段，并检查时间戳是否递增。缺少字段时记录错误并继续处理下一条记录。"
    ja = "パーサーは入力ファイルの各行を読み取り、フィールドに分割して、タイムスタンプが増加しているかを確認します。"
    ru = "Парсер читает каждую строку входного файла, разбивает её на поля и проверяет, что метка времени растёт. "
    digests = "\n".join(f"sha256 {rnd.getrandbits(256):064x}  build/artifact-{i}.tar" for i in range(400))
    refs = "\n".join(f"[Externalized tool output: tool=terminal; tool_call_id=call_{i}; chars=26800; bytes=26800; "
                     f"ref=20261119_101500_call_{i}_{rnd.getrandbits(48):012x}_{1763546100000000000 + i * 7919:x}.json]"
                     for i in range(200))
    return {"english_prose": fit(prose), "python_code": fit(code), "json_uuids": fit(js), "hex_trace_rows": hexrows,
            "base64_blob": fit(b64), "git_log_sha40": fit(git), "chinese": fit(zh), "japanese": fit(ja),
            "russian": fit(ru), "sha256_digests": fit(digests), "ns_ref_names": fit(refs)}


# name: (estimate before E'', E'')
EXPECTED = {
    "english_prose": (5001, 5001), "python_code": (5001, 5001), "json_uuids": (5001, 5001),
    "hex_trace_rows": (5001, 8379), "base64_blob": (5001, 5001), "git_log_sha40": (5001, 5001),
    "chinese": (13334, 13334), "japanese": (13334, 13334), "russian": (13334, 13334),
    "sha256_digests": (5001, 5001), "ns_ref_names": (5001, 5001),
}


@pytest.mark.parametrize("name", list(EXPECTED))
def test_hex_rule_gives_the_probe_values(name):
    text = samples()[name]
    before, after = EXPECTED[name]
    assert legacy_estimate(text) == before
    assert _fallback_token_estimate(text) == after
    if name != "hex_trace_rows":
        assert after == before


@pytest.mark.parametrize("text, expected", [
    ("", 0),
    ("a" * 7, 2),
    ("0123456789abcdef" * 4, 17),                      # 64 hex: a digest, not counted denser
    ("0123456789abcdef" * 4 + "0", 33),                # 65 hex with a letter and a digit: 2 chars/token
    ("1" * 200, 51),                                   # digits only
    ("f" * 200, 51),                                   # letters only
    ("x=" + "0123456789ABCDEF" * 5 + " end", 42),      # 80 upper-case hex inside other text: 40 + 6/4
])
def test_hex_rule_boundaries(text, expected):
    assert _fallback_token_estimate(text) == expected


def test_non_ascii_text_is_unchanged():
    for text in ("é" + "0123456789abcdef" * 10, "日本語" * 50 + "0123456789abcdef" * 10, "Привет " * 30):
        assert _fallback_token_estimate(text) == legacy_estimate(text)


def test_tokenizer_installs_are_not_resized(monkeypatch):
    class Encoder:
        def encode(self, text, **_kw):
            return list(range(len(text) // 3))
    monkeypatch.setattr(T, "_encoder_ready", True)
    monkeypatch.setattr(T, "_encoder", Encoder())
    monkeypatch.setattr(T, "_encoder_generation", T._encoder_generation + 1)
    T._count_tokens_cached.cache_clear()
    text = samples()["hex_trace_rows"]
    assert T.count_tokens(text) == len(text) // 3
