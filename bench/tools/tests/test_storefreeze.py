from __future__ import annotations

import hashlib
import json

from bench.tools import storefreeze


def _make_store(path):
    path.mkdir()
    (path / "a.txt").write_text("alpha\n", encoding="utf-8")
    (path / "nested").mkdir()
    (path / "nested" / "b.bin").write_bytes(b"\x00\x01")


def test_freeze_is_compact_self_hashed_and_copy_verifies(tmp_path):
    source = tmp_path / "source"
    destination = tmp_path / "private-copy"
    manifest_path = tmp_path / "store.manifest.json"
    _make_store(source)

    manifest = storefreeze.write_manifest(source, manifest_path)

    assert len(manifest_path.read_text(encoding="utf-8").splitlines()) == 1
    unsigned = {key: value for key, value in manifest.items() if key != "self_sha256"}
    canonical = json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
    assert manifest["self_sha256"] == hashlib.sha256(canonical).hexdigest()
    assert storefreeze.verify_manifest(source, manifest_path)["ok"]
    assert storefreeze.copy_verified(source, destination, manifest_path)["ok"]
    assert (destination / "nested" / "b.bin").read_bytes() == b"\x00\x01"


def test_verify_reports_changed_missing_and_added_names(tmp_path, capsys):
    source = tmp_path / "source"
    manifest_path = tmp_path / "store.manifest.json"
    _make_store(source)
    storefreeze.write_manifest(source, manifest_path)

    (source / "a.txt").write_text("changed\n", encoding="utf-8")
    (source / "nested" / "b.bin").unlink()
    (source / "new.txt").write_text("new\n", encoding="utf-8")

    result = storefreeze.verify_manifest(source, manifest_path)

    assert result == {
        "added": ["new.txt"],
        "changed": ["a.txt"],
        "manifest_valid": True,
        "missing": ["nested/b.bin"],
        "ok": False,
    }
    assert storefreeze.main(["verify", str(source), str(manifest_path)]) == 1
    assert json.loads(capsys.readouterr().out) == result


def test_tampered_manifest_fails_self_hash(tmp_path):
    source = tmp_path / "source"
    manifest_path = tmp_path / "store.manifest.json"
    _make_store(source)
    manifest = storefreeze.write_manifest(source, manifest_path)
    manifest["files"][0]["size"] += 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = storefreeze.verify_manifest(source, manifest_path)

    assert not result["manifest_valid"]
    assert not result["ok"]


def test_thousand_file_store_round_trip(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    for index in range(1000):
        (source / f"{index:04d}.json").write_text(
            f'{{"index":{index}}}\n', encoding="utf-8"
        )
    manifest_path = tmp_path / "store.manifest.json"

    manifest = storefreeze.write_manifest(source, manifest_path)

    assert len(manifest["files"]) == 1000
    assert storefreeze.verify_manifest(source, manifest_path)["ok"]
