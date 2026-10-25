"""Build the catalog payload from an exact Git revision or a tracked checkout."""
import argparse
import ast
import fnmatch
import hashlib
import json
from pathlib import Path
import subprocess

VERSION = 1
ASSETS = Path(__file__).resolve().parents[1] / "release/payload-assets.txt"
# tools/plugin_guard.py:102-104 at NousResearch/hermes-agent@6f7a7991bb069db07ae74a479823ce8310f8c7e0
MAX_FILES, MAX_TOTAL_BYTES, MAX_FILE_BYTES = 400, 10 * 1024**2, 1024**2
FORBIDDEN = {"tests", "bench", "scripts", ".github", ".agents", "docs"}


def closure(root, paths, read):
    modules = {p.removesuffix(".py").replace("/", "."): p for p in paths if p.endswith(".py")}
    modules.update({p[:-12].rstrip("/").replace("/", "."): p for p in paths if p.endswith("__init__.py")})
    seen, pending = set(), ["__init__.py"]
    while pending:
        path = pending.pop()
        if path in seen:
            continue
        if path not in paths:
            raise ValueError(f"unresolved local import: {path}")
        seen.add(path)
        package = path.split("/")[:-1]
        for node in ast.walk(ast.parse(read(path), filename=path)):
            targets = []
            if isinstance(node, ast.ImportFrom) and node.level:
                if node.level > len(package) + 1:
                    raise ValueError(f"unresolved local import above root: {path}")
                base = package[:len(package) + 1 - node.level]
                if node.module:
                    base += node.module.split(".")
                    targets = [".".join(base)]
                    pending.extend(modules[t] for alias in node.names if (t := ".".join(base + [alias.name])) in modules)
                else:
                    targets = [".".join(base + [alias.name]) for alias in node.names if alias.name != "*"]
            elif isinstance(node, ast.ImportFrom) and node.module == "hermes_lcm":
                targets = [alias.name for alias in node.names if alias.name != "*"]
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
                targets = [name.removeprefix("hermes_lcm.") for name in names if name.startswith("hermes_lcm.")]
                targets += [name for name in names if name.split(".")[0] in modules]
                if isinstance(node, ast.ImportFrom):
                    pending.extend(modules[t] for target in targets for alias in node.names
                                   if (t := target + "." + alias.name) in modules)
            for target in targets:
                if target not in modules:
                    raise ValueError(f"unresolved local import: {path}: {target}")
                pending.append(modules[target])
                pending.extend(modules[prefix] for i in range(1, len(target.split(".")))
                               if (prefix := ".".join(target.split(".")[:i])) in modules)
    return seen


def build(source, out, manifest, assets=ASSETS):
    checkout = Path(source).is_dir()
    root = Path(source).resolve() if checkout else Path.cwd()

    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args])

    sha = git("rev-parse", "--verify", "HEAD" if checkout else source + "^{commit}").decode().strip()
    if checkout and git("status", "--porcelain", "--untracked-files=no"):
        raise ValueError("dirty tracked checkout refused")
    paths = set(git("ls-files", "-z").decode().split("\0")) if checkout else {
        row.split("\t", 1)[1] for row in git("ls-tree", "-rz", sha).decode().split("\0") if row}
    paths.discard("")
    symlinks = set() if checkout else {row.split("\t", 1)[1] for row in
        git("ls-tree", "-rz", sha).decode().split("\0") if row.startswith("120000 ")}

    def read(path):
        if path in symlinks or (checkout and (root / path).is_symlink()):
            raise ValueError(f"payload symlink refused: {path}")
        if checkout and not (root / path).resolve().is_relative_to(root):
            raise ValueError(f"payload symlink escapes tree: {path}")
        return (root / path).read_bytes() if checkout else git("show", f"{sha}:{path}")

    selected = closure(root, paths, read)
    for pattern in assets.read_text().splitlines():
        if not pattern or pattern.startswith("#"):
            continue
        matches = {p for p in paths if fnmatch.fnmatchcase(p, pattern)}
        if not matches:
            raise ValueError(f"asset allowlist matches no files: {pattern}")
        selected.update(matches)
    required = {"plugin.yaml", "LICENSE"} | {p for p in paths if p.startswith("skills/hermes-lcm/")}
    if required - selected:
        raise ValueError(f"unlisted runtime assets: {sorted(required - selected)}")
    if any(p.split("/")[0] in FORBIDDEN for p in selected):
        raise ValueError("forbidden payload path")
    data = {p: read(p) for p in sorted(selected)}
    if len(data) > MAX_FILES or sum(map(len, data.values())) > MAX_TOTAL_BYTES or any(
            len(value) > MAX_FILE_BYTES for value in data.values()):
        raise ValueError("payload exceeds host limit")
    tooling_sha = subprocess.check_output(["git", "-C", str(ASSETS.parent.parent), "rev-parse", "HEAD"]).decode().strip()
    result = dict(source_sha=sha, tooling_sha=tooling_sha, allowlist_sha256=hashlib.sha256(assets.read_bytes()).hexdigest(),
                  builder_version=VERSION, file_count=len(data),
                  total_bytes=sum(map(len, data.values())), files=[
                      dict(path=p, bytes=len(value), sha256=hashlib.sha256(value).hexdigest()) for p, value in data.items()])
    Path(out).mkdir(parents=True, exist_ok=False)
    for path, value in data.items():
        destination = Path(out) / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(value)
    Path(manifest).write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("source", "out", "manifest"):
        parser.add_argument("--" + option, required=True)
    args = parser.parse_args()
    build(args.source, args.out, args.manifest)
