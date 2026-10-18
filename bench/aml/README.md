# AML Add/Search gateway

This loopback-only HTTP gateway serves the Agent Memory Leaderboard (AML)
Add/Search API through the OmniMemEval LCM-X client. Each operation runs in a
killable subprocess. Adds atomically publish complete generations containing
LCM-X data and a durable replay ledger; replaying a request returns its original
response without ingesting it again. Interrupted generations remain private.

## Reference versions and retrieval

AML Cycle 2's reference deployment served LCM-X tag `v0.28.0`, commit
`709099b3`, with the OmniMemEval LCM-X client at commit
`162640314863db64b5af62138b5b81776516d1ff`:
[`scripts/client_factory/lcmx_client.py`](https://github.com/electricsheephq/OmniMemEval/blob/162640314863db64b5af62138b5b81776516d1ff/scripts/client_factory/lcmx_client.py).

The adapter pins config `fleet`, provider `fastembed`, model
`BAAI/bge-small-en-v1.5`, chunk policy `full`, delivery `full`, and top-k cap
`100`. It uses an offline model cache, and an in-process Python audit hook in
each worker rejects non-loopback connections and DNS lookups so that no model
is downloaded. The hook guards against accidental downloads; it is not an
OS-level network sandbox. There are no generative-model calls in Add
or Search; the per-Add session summary is deterministic and rule-based.

## HTTP contract

- `POST /aml/add`: `user_id`, `request_id`, `session_id`, and `messages`.
  Messages have role `user` or `assistant`, string `content`, and optional
  Unix-millisecond `timestamp`.
- `POST /aml/search`: `user_id`, `query`, optional positive `top_k` (default
  and maximum 100), and optional string-array `options`.
- `GET /aml/health`: unauthenticated `{"status":"ok"}`.
- `/aml/v1/add`, `/aml/v1/search`, and `/aml/v1/health` are equivalent aliases.

Add and Search accept `X-Api-Key`, `Authorization: Token`, or
`Authorization: Bearer`; scheme matching is case-insensitive and token
comparison is constant-time. POST requires one `Content-Length`, with no
transfer encoding; request and search-response limits are 30 MiB.

`AML_WORKERS` defaults to 4 concurrent admitted requests (the Cycle 2 Full
evaluation ran with `AML_WORKERS=16`). Capacity is reserved
before reading the body; exhaustion returns HTTP 429 with `Retry-After: 60`.
Operations for the same user serialize, including across worker processes.
`AML_REQUEST_TIMEOUT_SECONDS` defaults to 1800 (also the maximum); it includes
body reading and lock waits, and an overdue worker is killed.

## Run

Use a Python environment with the pinned client's and LCM-X's dependencies.
Load your token from your own secret store into `AML_ADDSEARCH_TOKEN`; do not
put it in command history. Set the remaining environment variables:

```bash
export AML_DB_ROOT=/path/to/evaluation-data
export AML_OMNIMEMEVAL_REPO=/path/to/OmniMemEval
export AML_EMBED_CACHE_SOURCE=/path/to/offline/fastembed-cache
export AML_PYTHON=python3  # optional; defaults to python3
bash bench/aml/run_aml_server.sh /path/to/lcm-x-tree
```

The launcher requires the database root, OmniMemEval tree, and nonempty token.
The cache source defaults to `.cache/fastembed` in the home directory and must contain
`models--qdrant--bge-small-en-v1.5-onnx-q`; startup copies it into the database
root. Put a public HTTPS tunnel in front of `127.0.0.1:8791` for AML access.

Run the synthetic tests from the LCM-X repository root:

```bash
"${AML_PYTHON:-python3}" -m pytest bench/aml/test_aml_server.py -q
```

Delete evaluation data within 30 days, including retained copies and backups.
