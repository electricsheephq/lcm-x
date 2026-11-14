"""Isolated engine/session helpers and an absolute host-input row oracle."""
from copy import deepcopy
from dataclasses import dataclass, field
import sys
from types import SimpleNamespace


def _row(text, stamp, role='user'):
    return {'role': role, 'content': text, 'timestamp': stamp}


def _rows(engine):
    return engine._store.get_session_messages('parent')


def _identities(engine):
    return [(r['role'], r['content'], r['observed_at'], r['session_id'], r['conversation_id'])
            for r in _rows(engine)]


def _capture(ctx, history, earlier='P2', newest='R2', head_stamp=40., event_stamp=50.):
    head, tail = _row(earlier, head_stamp), _row(newest, event_stamp)
    ctx.hooks['pre_llm_call'](session_id='parent', conversation_id='conv', user_message=newest,
                              conversation_history=[*history, head, tail])
    head.update(content=newest, timestamp=event_stamp)
    return [*history, head, _row('B', event_stamp + 1, 'assistant')]


class AbsoluteViolation(AssertionError):
    """Stored rows lose host user text or repeat an entire row identity."""


@dataclass
class Trace:
    """Accumulate immutable host observations across engine recreation."""
    user_texts: set = field(default_factory=set)
    ingests: list = field(default_factory=list)
    host_objects: list = field(default_factory=list)
    captures: list = field(default_factory=list)
    preserved: dict = field(default_factory=dict)

    def observe(self, messages):
        self.user_texts.update(row['content'] for row in messages if row['role'] == 'user')

    def capture(self, messages):
        self.observe(messages)
        trailing = []
        for row in reversed(messages):
            if row['role'] != 'user':
                break
            trailing.insert(0, row)
        if len(trailing) < 2 or any(trailing[0] is old for old in self.host_objects):
            return
        # Only a never-ingested retained head, later overwritten with the exact
        # captured newest text/time, permits a joined-row substitution.
        head, newest = trailing[0], trailing[-1]
        self.captures.append((head, '\n\n'.join(row['content'] for row in trailing),
                              head['timestamp'], newest['content'], newest['timestamp']))

    def stored(self, engine, messages, originals):
        self.observe(messages)
        self.host_objects.extend(originals)
        for head, joined, stamp, newest, event_stamp in self.captures:
            if (any(head is row for row in originals) and head['content'] == newest
                    and head['timestamp'] == event_stamp):
                self.preserved[('user', joined, stamp, 'parent', 'conv')] = (
                    'user', newest, event_stamp, 'parent', 'conv')
        self.ingests.append(dict(host=deepcopy(messages), texts=sorted(self.user_texts),
                                 rows=_identities(engine), preserved=dict(self.preserved)))

    def violations(self):
        result = []
        for index, stage in enumerate(self.ingests, 1):
            rows = stage['rows']
            missing = [text for text in stage['texts']
                       if not any(role == 'user' and text in content
                                  for role, content, *_ in rows)]
            duplicates = sorted({row for row in rows if rows.count(row) > 1}, key=repr)
            if missing or duplicates:
                result.append(dict(ingest=index, missing=missing, duplicates=duplicates))
        return result


def _open(module, directory, monkeypatch, trace=None):
    monkeypatch.setenv('HERMES_HOME', str(directory / 'home'))
    monkeypatch.setenv('LCM_DATABASE_PATH', str(directory / 'lcm.db'))
    monkeypatch.setenv('LCM_EMBEDDINGS_ENABLED', '0')
    monkeypatch.setenv('LCM_BACKGROUND_LEAF_PREP_ENABLED', '0')
    ctx = SimpleNamespace(hooks={})
    ctx.register_context_engine = lambda engine: setattr(ctx, 'engine', engine)
    ctx.register_hook = lambda name, callback: ctx.hooks.__setitem__(name, callback)
    module.register(ctx)
    engine = ctx.engine
    monkeypatch.setattr(sys.modules[engine.__module__], 'summarize_with_escalation',
                        lambda **_kwargs: ('Earlier turns.', 1))
    engine.on_session_start('parent', platform='cli', context_length=128000, conversation_id='conv')
    if trace is not None:
        hook = ctx.hooks['pre_llm_call']

        def capture(*args, **kwargs):
            trace.capture(kwargs.get('conversation_history', []))
            if isinstance(kwargs.get('user_message'), str):
                trace.user_texts.add(kwargs['user_message'])
            return hook(*args, **kwargs)

        ingest = engine.ingest

        def observe(messages, *args, **kwargs):
            host = deepcopy(messages)
            result = ingest(messages, *args, **kwargs)
            trace.stored(engine, host, messages)
            return result

        ctx.hooks['pre_llm_call'] = capture
        monkeypatch.setattr(engine, 'ingest', observe)
    return ctx
