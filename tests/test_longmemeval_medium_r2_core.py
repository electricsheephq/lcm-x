"""Round-2 guards for LongMemEval streaming and prepared iteration."""

from __future__ import annotations

import hashlib
import sys
from types import SimpleNamespace

import pytest

import benchmarking.longmemeval as lme


class _FakeJSONError(Exception):
    pass


class _OffsetValueError(ValueError):
    pass


@pytest.mark.parametrize(
    ("backend_error", "detail"),
    [
        (_OffsetValueError("builder failed at byte offset 41"), "byte offset 41"),
        (KeyError("map_key"), "map_key"),
        (TypeError("builder type mismatch"), "builder type mismatch"),
    ],
)
def test_streaming_backend_builder_errors_fail_closed(
    monkeypatch, backend_error, detail
):
    def _items(*_args, **_kwargs):
        raise backend_error
        yield  # pragma: no cover - makes this a lazy backend iterator

    monkeypatch.setitem(
        sys.modules,
        "ijson",
        SimpleNamespace(JSONError=_FakeJSONError, items=_items),
    )

    with pytest.raises(ValueError, match="invalid LongMemEval dataset JSON") as caught:
        list(lme._iter_dataset_rows(object()))

    assert detail in str(caught.value)
    assert caught.value.__cause__ is backend_error


def _question(question_id: str) -> lme.Question:
    return lme.Question(
        question_id=question_id,
        question_type="single-session-user",
        question="question",
        haystack_session_ids=[],
        haystack_sessions=[],
        answer_session_ids=[],
    )


def _prepared_dataset(tmp_path, question_ids: tuple[str, ...]) -> lme.PreparedDataset:
    for question_id in question_ids:
        (tmp_path / f"{question_id}.json").write_text("{}", encoding="utf-8")
    return lme.PreparedDataset(
        directory=tmp_path,
        dataset_label="m",
        source_sha256="0" * 64,
        manifest_sha256="1" * 64,
        question_count=len(question_ids),
        questions=tuple(
            {
                "question_id": question_id,
                "file": f"{question_id}.json",
                "sha256": hashlib.sha256(question_id.encode()).hexdigest(),
            }
            for question_id in question_ids
        ),
    )


def test_prepared_qid_preflight_rejects_missing_file_before_scoring(tmp_path):
    prepared = _prepared_dataset(tmp_path, ("q0", "q1"))
    (tmp_path / "q1.json").unlink()

    with pytest.raises(ValueError, match="prepared question file not found"):
        prepared.validate_question_ids()


def test_prepared_qid_preflight_rejects_extra_file_before_scoring(tmp_path):
    prepared = _prepared_dataset(tmp_path, ("q0", "q1"))
    (tmp_path / "extra.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="file set does not match manifest"):
        prepared.validate_question_ids()


def test_prepared_qid_preflight_does_not_hash_question_bytes(tmp_path, monkeypatch):
    prepared = _prepared_dataset(tmp_path, ("q0", "q1"))

    def _unexpected_hash(_path):
        raise AssertionError("preflight hashed bytes")

    monkeypatch.setattr(lme, "sha256_file", _unexpected_hash)
    prepared.validate_question_ids()


def test_prepared_iterator_verifies_checksum_at_consumption(tmp_path):
    prepared = _prepared_dataset(tmp_path, ("q0",))

    with pytest.raises(ValueError, match="prepared question checksum mismatch: q0.json"):
        list(prepared.iter_questions())


def test_prepared_iterator_rejects_question_id_mismatch(tmp_path):
    # Checksum-valid file whose embedded id differs from the manifest entry —
    # only reachable via manifest corruption, still fails closed at consumption.
    payload = b'{"question_id": "q-other"}'
    (tmp_path / "q0.json").write_bytes(payload)
    prepared = lme.PreparedDataset(
        directory=tmp_path,
        dataset_label="m",
        source_sha256="0" * 64,
        manifest_sha256="1" * 64,
        question_count=1,
        questions=(
            {
                "question_id": "q0",
                "file": "q0.json",
                "sha256": hashlib.sha256(payload).hexdigest(),
            },
        ),
    )

    with pytest.raises(ValueError, match="prepared question id mismatch: q0.json"):
        list(prepared.iter_questions())


def test_question_filename_reserves_template():
    with pytest.raises(ValueError, match="unsafe question_id"):
        lme._question_filename("_TEMPLATE")


@pytest.mark.parametrize("question_id", [".hidden", "...", ".q1"])
def test_question_filename_rejects_leading_dot_ids(question_id):
    # glob("*.json") skips dotfiles on POSIX, so a hidden prepared file would
    # spuriously fail the manifest file-set check at load — reject at prepare.
    with pytest.raises(ValueError, match="unsafe question_id"):
        lme._question_filename(question_id)


@pytest.mark.parametrize(
    "question_id",
    [
        "CON",
        "prn.txt",
        "AUX",
        "nul",
        "COM1",
        "com9.json",
        "LPT1",
        "lpt9.txt",
        "question:name",
        "question*name",
        "question?name",
        'question"name',
        "question<name",
        "question>name",
        "question|name",
    ],
)
def test_question_filename_rejects_cross_platform_unsafe_names(question_id):
    with pytest.raises(ValueError, match="unsafe question_id"):
        lme._question_filename(question_id)


def test_run_harness_cleans_question_db_when_evaluation_raises(tmp_path, monkeypatch):
    question = _question("q0")
    error = RuntimeError("evaluation failed")
    cleanup_calls = []

    def _evaluate(*_args, **_kwargs):
        raise error

    def _cleanup(tmp_dir, question_id):
        cleanup_calls.append((tmp_dir, question_id))

    monkeypatch.setattr(lme, "evaluate_question", _evaluate)
    monkeypatch.setattr(lme, "_cleanup_question_db", _cleanup)

    with pytest.raises(RuntimeError) as caught:
        lme.run_harness(
            [question],
            provider_name="stub",
            model="",
            tmp_dir=tmp_path,
            reuse_db_template=False,
        )

    assert caught.value is error
    assert cleanup_calls == [(tmp_path, "q0")]
