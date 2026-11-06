"""#1410: materialize() is safe when parallel workers build the same baseline tree."""
import os
import threading

from tests.differential import trees


def test_concurrent_workers_share_one_complete_tree(tmp_path):
    head = trees.resolve("HEAD")
    results, errors = [], []
    start = threading.Barrier(4)

    def worker():
        try:
            start.wait(10)
            results.append(trees.materialize("HEAD", tmp_path))
        except Exception as exc:  # noqa: BLE001 - surfaced by the assertion below
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(60)
    assert errors == [] and len(results) == 4
    assert {str(path) for path in results} == {str(tmp_path / head)}
    assert (tmp_path / head / "tests" / "differential" / "trees.py").is_file()
    assert sorted(os.listdir(tmp_path)) == [head]  # no staging dir left behind


def test_a_worker_that_loses_the_rename_uses_the_winners_tree(tmp_path, monkeypatch):
    head = trees.resolve("HEAD")
    real_replace = os.replace

    def winner_first(src, dst):
        os.makedirs(dst)
        (tmp_path / head / "winner").write_text("")
        return real_replace(src, dst)  # the destination is now a non-empty directory

    monkeypatch.setattr(trees.os, "replace", winner_first)
    assert trees.materialize("HEAD", tmp_path) == tmp_path / head
    assert sorted(os.listdir(tmp_path)) == [head]
    assert os.listdir(tmp_path / head) == ["winner"]
