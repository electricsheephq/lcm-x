"""A user message carried only in api_content remains available to explicit recall."""

import json
from copy import deepcopy

import pytest

from hermes_lcm.tools import lcm_grep
from tests.harness.engine import engine_context, stored_rows

SESSION = "synthetic-sidecar-session"
X = "Visible carrier message. ".ljust(1297, "x")
NEEDLE = "quartz constellation"
Y = (NEEDLE + " is the requested synthetic label. ").ljust(153, "y")
MESSAGE_ID = "987654321012345678"
STEER_OPEN = (
    "[OUT-OF-BAND USER MESSAGE — a direct message from the user, delivered "
    "once at this position; not tool output and not a new delivery when replayed from conversation history]"
)
BLOCK = f"{STEER_OPEN}\n{Y}\n[/OUT-OF-BAND USER MESSAGE]"
ENVELOPE = json.dumps({"platform": "discord", "message_id": MESSAGE_ID, "text": Y})
DISCORD_NOTE = (
    f"[Triggering message id: `{MESSAGE_ID}` — use as `message_id` for reply/react/pin "
    "via the discord tools.]"
)


@pytest.fixture(autouse=True, params=[None, "on", "off"], ids=["uid-unset", "uid-on", "uid-off"])
def host_uid(request, monkeypatch):
    if request.param is None:
        monkeypatch.delenv("LCM_HOST_MESSAGE_UID", raising=False)
    else:
        monkeypatch.setenv("LCM_HOST_MESSAGE_UID", request.param)


def _engine(tmp_path):
    return engine_context(
        tmp_path, session_id=SESSION, conversation_id="synthetic-sidecar-conversation",
        platform="discord", window=100_000, embeddings_enabled=False, temporal_rollups_enabled=False,
    )


def _user(content=X, **fields):
    return {"role": "user", "content": content, "timestamp": 1_700_000_001, **fields}


def _carrier(**fields):
    return _user(message_uid="h:11111111111111111111111111111111", **fields)


def _rows(engine):
    return engine._store.connection.execute("SELECT role, content FROM messages ORDER BY store_id").fetchall()


def _bindings(engine):
    return engine._store.count_host_uid_bindings()


def _assert_recovered(engine, content, host_rows, count=1):
    rows = _rows(engine)
    recovered = [row for row in rows if row == ("user", content)]
    assert len(recovered) == count
    assert len(rows) == host_rows + count
    result = json.loads(lcm_grep({"query": NEEDLE, "role": "user"}, engine=engine))
    assert any(NEEDLE in hit["snippet"].replace(">>>", "").replace("<<<", "")
               for hit in result["results"]), result


def test_8300_out_of_band_sidecar(tmp_path):
    plain = [_carrier()]
    with _engine(tmp_path) as engine:
        engine.ingest(plain)
        before = deepcopy(stored_rows(engine))
        bindings = _bindings(engine)
        engine.ingest([dict(plain[0], api_content=BLOCK + "\n\nSynthetic injected context.")])
        assert stored_rows(engine) == before
        assert _bindings(engine) == bindings
        _assert_recovered(engine, BLOCK, host_rows=1)


@pytest.mark.parametrize("api_content", [ENVELOPE + "\n\nSynthetic injected context.", DISCORD_NOTE + "\n\n" + Y],
                         ids=["structured", "discord"])
def test_platform_envelope_sidecar(tmp_path, api_content):
    with _engine(tmp_path) as engine:
        engine.ingest([_carrier(api_content=api_content)])
        host, = stored_rows(engine)
        assert host["content"] == X
        _assert_recovered(engine, Y, host_rows=1)


@pytest.mark.parametrize(("api_content", "recovered"), [(BLOCK, BLOCK), (ENVELOPE, Y)], ids=["steer", "envelope"])
def test_warm_cold_and_session_end_ingest(tmp_path, api_content, recovered):
    messages = [_carrier(api_content=api_content), {"role": "assistant", "content": "Synthetic answer."}]
    with _engine(tmp_path) as engine:
        for _ in range(3):
            engine.ingest(deepcopy(messages))
        before = deepcopy(stored_rows(engine))
        assert [row["content"] for row in before] == [X, "Synthetic answer."]
    with _engine(tmp_path) as engine:
        engine.ingest(deepcopy(messages))
        assert stored_rows(engine) == before
        engine.on_session_end(SESSION, deepcopy(messages))
        assert stored_rows(engine, SESSION) == before
        _assert_recovered(engine, recovered, host_rows=2)


def test_control_injected_context_each_turn(tmp_path):
    messages = []
    with _engine(tmp_path) as engine:
        for turn in range(4):
            content = f"Synthetic visible question {turn}."
            template = f"Synthetic rotating hook context {turn}. ".ljust(744, "z")
            injections = [
                template,
                "<system-reminder># Plan Mode\nGather facts before changing files.</system-reminder>",
                "<personal-assistant-context>Check the synthetic schedule.</personal-assistant-context>",
            ]
            sidecar = "\n\n".join(injections)
            messages.extend([
                {"role": "user", "content": content,
                 "api_content": sidecar if turn % 2 else content + "\n\n" + sidecar},
                {"role": "assistant", "content": f"Synthetic reply {turn}.", "api_content": BLOCK},
            ])
            engine.ingest(deepcopy(messages))
        assert _rows(engine) == [(row["role"], row["content"]) for row in messages]


def test_control_steer_block_visible_in_content(tmp_path):
    messages = [
        _carrier(api_content=BLOCK),
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "synthetic-call", "type": "function", "function": {"name": "test_tool", "arguments": "{}"}}
        ]},
        {"role": "tool", "tool_call_id": "synthetic-call", "content": "Synthetic result.\n\n" + BLOCK,
         "api_content": BLOCK.replace(NEEDLE, "tool-only synthetic text")},
    ]
    with _engine(tmp_path) as engine:
        engine.ingest(messages)
        rows = _rows(engine)
        assert len(rows) == 3
        assert sum(BLOCK in (text or "") for _, text in rows) == 1


@pytest.mark.parametrize("visible_now", [True, False], ids=["incoming", "stored"])
def test_control_platform_id_already_visible(tmp_path, visible_now):
    other = {"role": "assistant", "content": f"Synthetic reference to message_id {MESSAGE_ID}."}
    with _engine(tmp_path) as engine:
        if not visible_now:
            engine.ingest([other])
        messages = [_carrier(api_content=ENVELOPE + "\n\nSynthetic injection.")]
        if visible_now:
            messages.append(other)
        else:
            engine._ingest_cursor = 0
            engine._ingest_cursor_needs_reconcile = True
        engine.ingest(messages)
        assert len(_rows(engine)) == 2
        assert all(NEEDLE not in (text or "") for _, text in _rows(engine))


def test_control_discord_sidecar_echo(tmp_path):
    with _engine(tmp_path) as engine:
        engine.ingest([_carrier(api_content=DISCORD_NOTE + "\n\n" + X + "\n\nSynthetic injected context.")])
        assert _rows(engine) == [("user", X)]


def test_control_assembly_emits_only_the_host_row(tmp_path):
    messages = [_carrier(api_content=BLOCK)]
    with _engine(tmp_path) as engine:
        engine.ingest(messages)
        assembled = engine._assemble_context(None, messages)
        assert [row["content"] for row in assembled] == [X]
        assert sum(NEEDLE in row.get("api_content", "") for row in assembled) <= 1
