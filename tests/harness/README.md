# Behavioral test harness

Use `from tests.harness import ...` in tests that exercise the real LCM engine.
Fixtures are explicit: importing the package does not patch or start anything.
Migrate existing tests only when a PR already touches them; no bulk migration.

- `make_engine(tmp_path, *, window=None, session_id="S", conversation_id=None,
  platform="cli", **overrides)` uses fresh `LCMConfig()` defaults, an isolated
  `lcm.db`, and engine `hermes_home=tmp_path/"home"`. Unknown config names raise
  `AttributeError`. `database_path` cannot be overridden and raises `TypeError`.
  Only a supplied window starts the session.
- `engine_context(...)` accepts the same arguments and always shuts down.
- `pin_char_counter(monkeypatch)` pins `_encoder_ready=True`, `_encoder=None` and
  `_get_encoder` (so an encoder load still running from an earlier test cannot switch
  counters), isolates the token cache, and disables the host estimator. This uses the real
  fallback through imported aliases, as in the estimate case of
  `test_issue_947_clipped_no_verbatim.py` and the offline differential drivers;
  host-estimator isolation follows `test_issue_628_summary_circuit.py`.
  Install before constructing engines or scheduling work.
- `ScriptedSummarizer(mode="healthy", replies=None, fn=None).install(monkeypatch)`
  patches only `escalation._call_llm_for_summary`; circuit/leaf/log paths stay real.
  Healthy text uses the accepted summary plus expansion footer from
  `test_issue_628_summary_circuit.py`; this seam returns already-validated text.
  `outage` returns `None`; `scripted` repeats the last reply or calls `fn` with
  provider arguments. `.calls` records `(model, max_tokens, prompt_length)`.
- `user(text)`, `assistant(text, calls=())`, `call(id, name="probe", arguments="{}")`,
  and `tool(call_id, text, name="probe")` build Hermes 0.21.x rows, following
  `test_issue_1220_survival_short_rows.py` (including both tool-name fields).
- `sequential_pairs(n, chars)` starts with one user request, then n assistant/tool
  pairs, each with `chars` characters of assistant prose and tool content.
  `parallel_batch(n, chars)` has one user, one empty assistant with n calls,
  and n results of `chars` characters. Call IDs are unique within each view.
- `prose(chars, seed=0, *, sentinel="")` returns exactly `chars` seeded characters,
  including the sentinel prefix; a prefix that cannot fit raises `ValueError`.
- `host_call(engine, view, *, retry_on_overflow=True)` ingests, counts, checks
  pressure as the host does (`should_compress`, then, below the threshold and unblocked,
  `should_compress_preflight`), compresses, then retries once with cooldown bypass on
  window overflow.
  It returns `HostCall(view, sent_tokens, compressed, retried, overflow_failed)`;
  `compressed` means compression was invoked (possibly a no-op). Zero is an
  unknown window. Counts use the configured counter; pin it for offline tests.
- `stored_rows(engine, session_id=None)` reads the current or specified session
  through the store's public accessor.
Install both counter and summarizer fakes for deterministic offline scenarios.
