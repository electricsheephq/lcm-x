"""Single refactor seam for the compatibility suite's private codec entry points."""

import time
from types import SimpleNamespace


def deterministic_runtime(monkeypatch):
    from hermes_lcm import externalize, survival_fit, tokens

    monkeypatch.setattr(tokens, "_encoder", None)
    monkeypatch.setattr(tokens, "_encoder_ready", True)
    tokens._count_tokens_cached.cache_clear()
    monkeypatch.setattr(survival_fit, "_host_estimate", lambda messages: None)
    # Freeze payload names, not scheduler or timeout clocks.
    monkeypatch.setattr(externalize, "time", SimpleNamespace(
        strftime=time.strftime, gmtime=lambda: time.gmtime(1700000000),
        time_ns=lambda: 1700000000000000000, time=lambda: 1700000000.0))


def clear_token_cache():
    from hermes_lcm import tokens

    tokens._count_tokens_cached.cache_clear()


def projected_fields(engine, row, source, *, stored_calls=False):
    from hermes_lcm.tokens import count_message_tokens

    return engine._survival_projected_fields(row, count_message_tokens(source), 1200, 600,
                                             mark_calls=stored_calls)


def measure(engine, view):
    return engine._survival_measure(view)


def project(engine, source, store_id, limit):
    return engine._survival_projection([source], {id(source): store_id}, limit)[0]


def is_mark_only(text):
    from hermes_lcm import survival_fit

    return survival_fit._PROJECTED_RE.fullmatch(text) is not None


def fit(engine, view, limit):
    return engine._survival_fit(view, view, measure(engine, view), "compat_fixture", request_cap=limit)


# Executed only after the subprocess imports its pinned reader, never this tree's codec.
OLD_RUNTIME_SETUP = '''
tokens._encoder, tokens._encoder_ready = None, True
tokens._count_tokens_cached.cache_clear()
survival_fit._host_estimate = lambda messages: None
'''
OLD_FIT = '''
        fitted = engine._survival_fit(view, view, engine._survival_measure(view), "compat_fixture")
'''
