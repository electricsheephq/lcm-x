#!/usr/bin/env python3
"""Freeze, verify, and privately copy file stores."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def snapshot(directory: str | Path) -> dict[str, Any]:
    root = Path(directory)
    if not root.is_dir():
        raise NotADirectoryError(root)
    files = []
    for path in sorted(path for path in root.rglob("*") if path.is_file()):
        files.append(
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": sha256_file(path),
                "size": path.stat().st_size,
            }
        )
    body = {"algorithm": "sha256", "files": files, "version": 1}
    return {**body, "self_sha256": hashlib.sha256(_canonical(body)).hexdigest()}


def write_manifest(directory: str | Path, output: str | Path) -> dict[str, Any]:
    manifest = snapshot(directory)
    Path(output).write_text(_canonical(manifest).decode() + "\n", encoding="utf-8")
    return manifest


def load_manifest(path: str | Path) -> dict[str, Any]:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("manifest is not a JSON object")
    return manifest


def verify_manifest(directory: str | Path, manifest_path: str | Path) -> dict[str, Any]:
    expected = load_manifest(manifest_path)
    unsigned = {key: value for key, value in expected.items() if key != "self_sha256"}
    expected_self = expected.get("self_sha256")
    manifest_valid = expected_self == hashlib.sha256(_canonical(unsigned)).hexdigest()

    current = snapshot(directory)
    expected_files = {item["path"]: item for item in expected.get("files", [])}
    current_files = {item["path"]: item for item in current["files"]}
    shared = expected_files.keys() & current_files.keys()
    changed = sorted(
        path
        for path in shared
        if (
            expected_files[path].get("sha256"),
            expected_files[path].get("size"),
        )
        != (
            current_files[path].get("sha256"),
            current_files[path].get("size"),
        )
    )
    return {
        "added": sorted(current_files.keys() - expected_files.keys()),
        "changed": changed,
        "manifest_valid": manifest_valid,
        "missing": sorted(expected_files.keys() - current_files.keys()),
        "ok": manifest_valid
        and not changed
        and current_files.keys() == expected_files.keys(),
    }


def copy_verified(
    source: str | Path,
    destination: str | Path,
    manifest_path: str | Path,
) -> dict[str, Any]:
    before = verify_manifest(source, manifest_path)
    if not before["ok"]:
        raise ValueError(f"source does not match manifest: {json.dumps(before)}")
    shutil.copytree(source, destination)
    after = verify_manifest(destination, manifest_path)
    if not after["ok"]:
        raise ValueError(f"copy does not match manifest: {json.dumps(after)}")
    return after


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Freeze and verify file stores with compact sha256 manifests."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    freeze = subparsers.add_parser("freeze", help="write a compact store manifest")
    freeze.add_argument("directory", type=Path)
    freeze.add_argument("-o", "--output", type=Path)

    verify = subparsers.add_parser("verify", help="verify exact names, sizes, and shas")
    verify.add_argument("directory", type=Path)
    verify.add_argument("manifest", type=Path)

    copy = subparsers.add_parser(
        "copy-verified", help="copy a verified source to a new private directory"
    )
    copy.add_argument("source", type=Path)
    copy.add_argument("destination", type=Path)
    copy.add_argument("manifest", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "freeze":
        output = args.output or args.directory.with_name(
            f"{args.directory.name}.manifest.json"
        )
        manifest = write_manifest(args.directory, output)
        print(
            json.dumps(
                {
                    "files": len(manifest["files"]),
                    "manifest": str(output),
                    "self_sha256": manifest["self_sha256"],
                },
                sort_keys=True,
            )
        )
        return 0
    if args.command == "verify":
        result = verify_manifest(args.directory, args.manifest)
        print(json.dumps(result, sort_keys=True))
        return 0 if result["ok"] else 1

    result = copy_verified(args.source, args.destination, args.manifest)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
