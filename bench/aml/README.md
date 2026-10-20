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

### Search mode

`AML_SEARCH_MODE` selects `textual` (default when unset or empty) or `coding`
at startup; other values prevent startup. Coding mode bounds return size for
the Coding track with fixed `CODING_TOP_K = 10`, `CODING_ITEM_CHARS = 4000`,
and `CODING_TOTAL_CHARS = 40000`. It preserves ranked order and full-content
IDs while returning excerpts through the same endpoints. Use a separate
`AML_DB_ROOT` per mode.

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

## Opt-in v2 coding delivery

Set `AML_V2_DB_ROOT=/path/to/separate-v2-data` to enable `/aml/v2/add`,
`/aml/v2/search`, and `/aml/v2/health` (GET or HEAD). Roots must be disjoint.
The frozen routes retain their mode and per-request workers. V2 uses the same
adapter retrieval with verbatim code, role/time/trajectory headers and matched
windows: 4,000 chars per hit, spare budget up to 10,240, at most 10 hits and
40,000 content chars with stable stored-row IDs and duplicate rows removed.
V2 uses the same killable per-request workers as the frozen routes. Health is
200 once its root/cache is prepared; a v2 preparation failure returns 503 on v2
routes while frozen routes remain available. Adds retain atomic generations
and user serialization. Both route families append metadata-only
receipts to `AML_RECEIPTS` (default `<AML_DB_ROOT>/receipts.jsonl`); receipts
contain no queries, content or tokens; write failures never affect responses.
The product's embedding thread constant is unchanged; this adapter exposes no
thread configuration knob.

AML binds one endpoint URL per leaderboard key, so a new version cannot point
at `/aml/v2/*`. Set `AML_V1_ROUTES=v2` (with `AML_V2_DB_ROOT`) to serve the
bound `/aml/v1/add`, `/aml/v1/search` and `/aml/v1/health` with v2 delivery;
the default `frozen` keeps them on the frozen routes. Other values, or `v2`
without `AML_V2_DB_ROOT`, prevent startup. Receipts keep the incoming route.
Register a new version before switching, because the delivery changes.

## Private request capture (debugging only)

Set `AML_CAPTURE_DIR=/path/to/private-dir` to append one JSON line per
successful add or search to `capture.jsonl` in that directory, written after
the reply is sent. Search lines hold the query, the requested `top_k` and the
exact returned `data`; add lines hold only each item's size and SHA-256, never
its content. The directory must be a private directory owned by the service
user: it is created 0700 when missing, and a link or any group or other access
disables capture. The file is created 0600 and must be a regular file (never a
link or FIFO). A capture failure never changes a response and logs only the
exception type. Unset, nothing is captured.

Captured lines are evaluation data. Use them only to debug that evaluation,
keep them off shared or synced storage, and delete them with the rest of the
evaluation data, including any copies made for analysis.
