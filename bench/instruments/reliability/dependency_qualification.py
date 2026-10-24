"""Exact-host dependency-carrier cells (a)-(f) for #1186 S1-C.2.

Run only on isolated prepared hosts. Every subprocess has raw stdout/stderr and
command metadata saved before assertions; failed runs retain partial receipts.
The JSON measurements are raw inputs, not citable benchmark cards.
"""
from __future__ import annotations

import argparse
import errno
import json
import os
from pathlib import Path
import pty
import re
import select
import shutil
import signal
import subprocess
import time

REPO = Path(__file__).resolve().parents[3]
PROBE = Path(__file__).with_name("dependency_probe.py")
CORE = "hermes-lcm-x"
COMPANION = "hermes-lcm-x-local-embeddings"
SOURCE = "electricsheephq/lcm-x/companions/" + COMPANION
GIB = 1024**3
ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def allocated_bytes(root: Path) -> int:
    """Allocated blocks, counting a hard-linked inode once within this tree."""
    seen, total = set(), 0
    for directory, _, files in os.walk(root):
        for name in files:
            stat = (Path(directory) / name).lstat()
            inode = stat.st_dev, stat.st_ino
            if inode not in seen:
                total += stat.st_blocks * 512
                seen.add(inode)
    return total


def answer_prompt(prompt: str, *, decline: bool) -> str:
    """Test user input is sent only after an actual host y/N prompt appears."""
    return "n" if decline and "Prepare these with Hermes through PM now?" in prompt else "y"


class Qualification:
    def __init__(self, host: str, python: Path, src: Path, scratch: Path, out: Path,
                 *, route: str = "native", allow_locked_tools: bool = False):
        self.host, self.python, self.src = host, python, src
        self.scratch, self.out = scratch, out
        self.pm = host == "upstream-main"
        self.route = route
        self.allow_locked_tools = allow_locked_tools
        self.sequence = 0
        self.cells = []
        self.out.mkdir(parents=True, exist_ok=True)
        self.scratch.mkdir(parents=True, exist_ok=False)
        self.head = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
        self.initial_bytes = allocated_bytes(scratch)
        self.fixture = scratch / "core-source"

    def env(self, home: Path) -> dict:
        # No ambient provider credentials, user configuration or real model cache.
        return {
            "HOME": str(self.scratch / "native-home"), "HERMES_HOME": str(home),
            "PATH": os.pathsep.join((str(self.python.parent), os.environ.get("PATH", "/usr/bin:/bin"))),
            "PYTHONPATH": str(self.src), "PYTHONDONTWRITEBYTECODE": "1",
            # Native dependency installation must use the host's normal policy;
            # a sealed legacy venv would refuse the requested plugin install.
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1", "FASTEMBED_CACHE_PATH": str(self.scratch / "models-empty"),
            "XDG_CACHE_HOME": str(self.scratch / "cache"), "UV_CACHE_DIR": str(self.scratch / "uv-cache"),
            "LCM_EMBEDDINGS_ENABLED": "true", "LCM_EMBEDDING_PROVIDER": "fastembed",
            "LCM_EMBEDDING_MODEL": "BAAI/bge-small-en-v1.5", "LCM_RERANK_ENABLED": "false",
            "NO_COLOR": "1", "TERM": "dumb", "COLUMNS": "240",
            "LCM_1186_HOSTED": "1" if os.environ.get("GITHUB_ACTIONS") == "true" else "0",
        }

    def guard(self) -> None:
        free = shutil.disk_usage(self.scratch).free
        assert free >= 40 * GIB, f"DISK_STOP: {free} bytes free; requires at least {40 * GIB} bytes (40 GiB)"
        assert allocated_bytes(self.scratch) - self.initial_bytes < 15 * GIB, "DISK_STOP: 15 GiB peak cap"

    def run(self, home: Path, command: list[str], *, tty: bool = False, decline: bool = False,
            timeout: int = 600) -> str:
        self.guard()
        self.sequence += 1
        prefix = self.out / f"{self.sequence:03d}"
        started = time.monotonic()
        raw, err, prompts = b"", b"", []
        status = "RUNNING"
        process = None
        try:
            if tty:
                master, slave = pty.openpty()
                process = subprocess.Popen(command, cwd=self.src, env=self.env(home),
                                           stdin=slave, stdout=slave, stderr=slave, start_new_session=True)
                os.close(slave)
                answered = 0
                try:
                    while True:
                        if time.monotonic() - started > timeout:
                            raise subprocess.TimeoutExpired(command, timeout)
                        ready, _, _ = select.select([master], [], [], 0.2)
                        if ready:
                            try:
                                chunk = os.read(master, 65536)
                            except OSError as exc:
                                if exc.errno != errno.EIO:
                                    raise
                                chunk = b""
                            if not chunk:
                                break
                            raw += chunk
                            clean = ANSI.sub("", raw.decode(errors="replace"))
                            pending = list(re.finditer(r"[^\r\n]*\[y/N\]:\s*", clean))
                            for match in pending[answered:]:
                                prompt = match.group()
                                answer = answer_prompt(prompt, decline=decline)
                                prompts.append({"prompt": prompt, "answer": answer})
                                os.write(master, (answer + "\n").encode())
                                answered += 1
                        elif process.poll() is not None:
                            break
                    process.wait(timeout=10)
                finally:
                    os.close(master)
            else:
                process = subprocess.Popen(command, cwd=self.src, env=self.env(home),
                                           stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                           start_new_session=True)
                raw, err = process.communicate(timeout=timeout)
            status = "COMPLETE" if process.returncode == 0 else "VERIFIED_FAILURE"
        except subprocess.TimeoutExpired:
            status = "STOP_REBUILD_OVER_10_MINUTES"
            if process is not None:
                os.killpg(process.pid, signal.SIGKILL)
                tail, stderr = process.communicate()
                raw += tail or b""
                err += stderr or b""
            raise
        finally:
            prefix.with_suffix(".stdout").write_bytes(raw)
            prefix.with_suffix(".stderr").write_bytes(err)
            prefix.with_suffix(".json").write_text(json.dumps({
                "command": command, "home": str(home), "tty": tty, "prompts": prompts,
                "status": status, "returncode": process.returncode if process else None,
                "wall_seconds": time.monotonic() - started,
            }, indent=2) + "\n")
        assert process.returncode == 0, f"{prefix}: {raw[-3000:]!r} {err[-3000:]!r}"
        self.guard()
        return ANSI.sub("", raw.decode(errors="replace"))

    def cli(self, home: Path, *args: str, **kwargs) -> str:
        return self.run(home, [str(self.python), "-m", "hermes_cli.main", *args], **kwargs)

    def probe(self, home: Path, mode: str) -> dict:
        python, arguments = self.python, []
        if self.pm and mode in {"import", "fts"}:
            # Read selection through the exact bootstrap interpreter; launch
            # product behavior in the actual committed generation, never a
            # foreign venv carrying only the prep environment's dependencies.
            selected = self.probe(home, "state")
            assert selected["committed"] and selected["stamp"], (
                "native core admission must commit a PM generation before runtime probing", selected)
            python = Path(selected["python"])
            assert python.is_file(), selected
            arguments = [selected["environment"]]
        text = self.run(home, [str(python), str(PROBE), mode, *arguments])
        return json.loads(text.strip().splitlines()[-1])

    def record(self, cell: str, **evidence) -> None:
        self.cells.append({"cell": cell, "host": self.host, "status": "COMPLETE", **evidence})
        self.save()

    def save(self) -> None:
        (self.out / "cells.json").write_text(json.dumps({
            "host": self.host, "candidate_sha": self.head, "cells": self.cells,
            "measurement_status": "raw; result card and publication required before citation",
        }, indent=2) + "\n")

    def preflight(self, home: Path) -> None:
        pins = json.loads((REPO / "bench/instruments/reliability/hosts.ci.json").read_text())["hosts"]
        spec = pins[self.host]
        actual = self.run(home, ["git", "-C", str(self.src), "rev-parse", "HEAD"]).strip()
        assert actual == spec["sha"], (actual, spec["sha"])
        version = self.run(home, [str(self.python), "-c", "import sys; print('.'.join(map(str,sys.version_info[:2])))"]).strip()
        assert version == spec["python_version"], (version, spec["python_version"])
        if self.pm:
            if self.allow_locked_tools:
                assert os.environ.get("GITHUB_ACTIONS") == "true", "locked archives: isolated hosted CI only"
                self.record("locked-tools", **self.probe(home, "locked-tools"))
            ready = self.probe(home, "preflight")["locked_tools_ready"]
            assert ready or self.allow_locked_tools, (
                "ENVIRONMENT_BLOCKED: PM requires lock-verified uv/Python archives; "
                "PATH tools cannot satisfy pm._uv._toolchain. No archive download authorized."
            )
        self.record("identity", host_sha=actual, python_version=version)

    def fixture_source(self, home: Path) -> None:
        self.run(home, ["git", "clone", "--quiet", "--no-hardlinks", str(REPO), str(self.fixture)])
        self.run(home, ["git", "-C", str(self.fixture), "checkout", "-b", "qualification-update"])

    def install(self, home: Path, route: str, *, tty: bool = True, decline: bool = False) -> str:
        home.mkdir(parents=True, exist_ok=True)
        if route == "installer":
            text = self.run(home, ["bash", str(REPO / "scripts/install.sh")], tty=tty, decline=decline)
        else:
            # This public host source type permits a deterministic local Git
            # upstream for the subsequent real unattended-update transaction.
            # S1-C.3 allows scanner-off for this core native install only.
            # Restore before EVERY companion install, including decline cells.
            (home / "config.yaml").write_text("plugins:\n  scan_on_install: false\n")
            try:
                self.cli(home, "plugins", "install", self.fixture.as_uri(), tty=True)
            finally:
                self.run(home, [str(self.python), "-c",
                                "from hermes_cli.config import load_config, save_config; "
                                "c=load_config(); c.setdefault('plugins', {})['scan_on_install']=True; save_config(c)"])
            scanned = self.run(home, [str(self.python), str(PROBE), "scan",
                                     str(self.fixture / "companions" / COMPANION)])
            scan = json.loads(scanned.strip().splitlines()[-1])
            assert scan["scanner_enabled"] and scan["verdict"] == "safe", scan
            self.record("companion-scan", route=route, **scan)
            text = self.cli(home, "plugins", "install", SOURCE, "--ref", self.head, tty=tty, decline=decline)
        scanned = self.run(home, [str(self.python), str(PROBE), "scan",
                                 str(home / "plugins" / COMPANION)])
        scan = json.loads(scanned.strip().splitlines()[-1])
        assert scan["scanner_enabled"] and scan["verdict"] == "safe", scan
        self.record("companion-scan", route=route, **scan)
        assert (home / "plugins" / COMPANION / "plugin.yaml").is_file()
        assert not (home / "plugins" / COMPANION).is_symlink()
        return text

    def fresh_routes(self) -> Path:
        home = self.scratch / self.route
        for route in (self.route,):
            text = self.install(home, route)
            result = self.probe(home, "import")
            assert result["companion_enabled"] and not result["companion_error"], result
            if self.pm:
                assert "Prepare these with Hermes through PM now?" in text, text
            self.record("a", route=route, **result)
        return home

    def decline_routes(self) -> None:
        if not self.pm:
            for cell in ("c", "d", "e", "f"):
                self.cells.append({"cell": cell, "host": self.host, "status": "N/A",
                                   "reason": "reference pin has legacy dependency installation, no PM consent/generations"})
            self.save()
            return
        for route, tty in ((self.route, self.route == "native"),):
            home = self.scratch / ("decline" if tty else "noninteractive")
            text = self.install(home, route, tty=tty, decline=True)
            assert "hermes plugins enable" in text, text
            result = self.probe(home, "fts")
            assert not result["companion_enabled"], result
            self.record("d", route=route, **result)
            if not tty:
                command = f"hermes plugins enable {COMPANION}"
                assert command in text, text
                self.cli(home, "plugins", "enable", COMPANION)
                enabled = self.probe(home, "import")
                assert enabled["companion_enabled"], enabled
                self.record("f", printed_command=command, **enabled)

    def profiles(self, home: Path) -> None:
        before = self.probe(home, "state")
        observations = [self.probe(home, "import")]
        for name in ("qualification-one", "qualification-two"):
            self.cli(home, "profile", "create", name, "--no-alias", "--no-skills")
            profile = home / "profiles" / name
            (profile / "plugins").mkdir(exist_ok=True)
            for plugin in (CORE, COMPANION):
                (profile / "plugins" / plugin).symlink_to(home / "plugins" / plugin)
            # Real profiles have independent selections; explicit host enables
            # admit their default-home managed plugin sources into the union.
            self.cli(profile, "plugins", "enable", CORE)
            self.cli(profile, "plugins", "enable", COMPANION)
            observations.append(self.probe(profile, "import"))
            after = self.probe(profile, "state")
            assert after == before, (before, after)
        assert self.probe(home, "state") == before
        self.record("c", generation=before, observations=observations)

    def unattended_update(self, home: Path) -> None:
        before = self.probe(home, "state") if self.pm else None
        manifest = self.fixture / "plugin.yaml"
        manifest.write_text(re.sub(r"(?m)^version:.*$", "version: 999.1186.1", manifest.read_text()))
        self.run(home, ["git", "-C", str(self.fixture), "-c", "user.name=Qualification Fixture",
                        "-c", "user.email=fixture@example.invalid", "commit", "-am", "Synthetic core version bump"])
        target = home / "plugins" / CORE
        code = (
            "from pathlib import Path; "
            "from hermes_cli import plugins_cmd_update as pc; "
            f"print(pc._pull_plugin_update(Path({str(target)!r}), 'pinned', 'not git', interactive=False))"
            if self.pm else
            "from pathlib import Path; from hermes_cli import plugins_cmd as pc; "
            f"print(pc._pull_plugin_update(Path({str(target)!r}), 'pinned', 'not git'))"
        )
        if self.pm:
            # S1-C.4: retain the default-scanner refusal beside the exception
            # arm. This is the real update transaction, before publication.
            assert os.environ.get("GITHUB_ACTIONS") == "true", "update scan exception: hosted CI only"
            assert home.resolve().is_relative_to(self.scratch.resolve()), "synthetic home only"
            refusal = (
                "import json; from hermes_cli.plugins_cmd import PluginScanBlocked, _scan_on_install_enabled; "
                "assert _scan_on_install_enabled();\n"
                "try:\n " + code + "\n"
                "except PluginScanBlocked:\n print(json.dumps({'scanner_enabled': True, 'update_refused': True}))\n"
                "else:\n raise AssertionError('expected full-core scanner refusal; revisit exception')"
            )
            refused = json.loads(self.run(home, [str(self.python), "-c", refusal]).strip().splitlines()[-1])
            assert refused["scanner_enabled"] and refused["update_refused"], refused
            assert "999.1186.1" not in (target / "plugin.yaml").read_text()
            assert self.probe(home, "state") == before
            self.record("b-scanner-on-refusal", status="EXPECTED_NEGATIVE", **refused)
            setting = (
                "from hermes_cli.config import load_config, save_config; "
                "c=load_config(); c.setdefault('plugins', {})['scan_on_install']={enabled}; save_config(c)"
            )
            try:
                self.run(home, [str(self.python), "-c", setting.replace("{enabled}", "False")])
                self.run(home, [str(self.python), "-c", code])
            finally:
                self.run(home, [str(self.python), "-c", setting.replace("{enabled}", "True")])
            scanned = self.run(home, [str(self.python), str(PROBE), "scan",
                                     str(home / "plugins" / COMPANION)])
            scan = json.loads(scanned.strip().splitlines()[-1])
            assert scan["scanner_enabled"] and scan["verdict"] == "safe", scan
            self.record("companion-scan", route="after-core-update", **scan)
        else:
            self.run(home, [str(self.python), "-c", code])
        assert "999.1186.1" in (target / "plugin.yaml").read_text()
        result = self.probe(home, "import")
        after = self.probe(home, "state") if self.pm else None
        assert after == before, (before, after)
        self.record("b", interactive=False, before=before, after=after, **result)
        if self.pm:
            self.record("e-core-version-no-rebuild", update_cell="b", before=before, after=after)

    def rebuilds(self, home: Path) -> None:
        before = self.probe(home, "state")
        leases, retained, measurements = [], [], []
        try:
            for bound in ("0.99", "0.98"):
                env = Path(before["environment"])
                lease = subprocess.Popen([str(self.python), str(PROBE), "lease"], cwd=self.src,
                                         env=self.env(home), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                         stderr=subprocess.PIPE, text=True)
                leases.append(lease)
                ready, _, _ = select.select([lease.stdout], [], [], 30)
                assert ready and lease.stdout.readline().strip() == "LEASE_READY", "lease process failed"
                retained.append(str(env))
                manifest = home / "plugins" / COMPANION / "plugin.yaml"
                manifest.write_text(re.sub(r"fastembed>=0\.8\.0,<[^\"'\]\s]+", f"fastembed>=0.8.0,<{bound}",
                                           manifest.read_text()))
                generations = Path(before["state_dir"]) / "environments"
                names_before = {p.name for p in generations.iterdir() if p.is_dir()}
                bytes_before = allocated_bytes(self.scratch)
                started = time.monotonic()
                # Native installation and real dependency consent above are
                # prerequisites. This public normal-sync API measures only the
                # already-consented carrier's range edit, without a CLI enable
                # no-op or adding an unrelated host feature to the graph.
                self.run(home, [str(self.python), "-c",
                                "from pm import sync_venv; sync_venv(explicit=True); print('SYNC_COMPLETE')"])
                elapsed = time.monotonic() - started
                after = self.probe(home, "state")
                assert after["stamp"] != before["stamp"] and after["environment"] != before["environment"]
                created = {p.name for p in generations.iterdir() if p.is_dir()} - names_before
                assert len(created) == 1, created
                generation_bytes = allocated_bytes(Path(after["environment"]).parent)
                row = {"range_upper_bound": bound, "wall_seconds": elapsed,
                       "allocated_generation_bytes": generation_bytes,
                       "allocated_delta_bytes": allocated_bytes(self.scratch) - bytes_before,
                       "new_generation_names": sorted(created),
                       "before": before, "after": after, "retained_leased_environments": list(retained),
                       "leases": self.probe(home, "leases")}
                measurements.append(row)
                (self.out / "rebuild-measurements.json").write_text(json.dumps(measurements, indent=2) + "\n")
                assert elapsed <= 600 and generation_bytes <= 3 * GIB, "STOP_REBUILD_COST: exceeds 10 min / 3 GiB"
                assert all(Path(p).is_dir() for p in retained), row
                assert all(row["leases"].get(Path(p).parent.name) for p in retained), row
                self.probe(home, "import")
                before = after
            self.record("e", core_version_update_cell="b", measurements=measurements)
        finally:
            for lease in leases:
                lease.communicate(timeout=30)

    def execute(self) -> None:
        home = self.scratch / "native"
        self.preflight(home)
        self.fixture_source(home)
        scanned = self.run(home, [str(self.python), str(PROBE), "scan", str(self.fixture)])
        scan = json.loads(scanned.strip().splitlines()[-1])
        (self.out / "core-scanner-flags.json").write_text(json.dumps(scan, indent=2) + "\n")
        # Prove missing-provider behavior before any home consents to the
        # carrier; shared host selections must not pre-populate fastembed.
        self.decline_routes()
        native = self.fresh_routes()
        if self.route == "native":
            if self.pm:
                self.profiles(native)
                self.rebuilds(native)
            self.unattended_update(native)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", choices=["ref-0.21.5", "upstream-main"], required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--route", choices=["native", "installer"], default="native")
    parser.add_argument("--allow-locked-tools", action="store_true",
                        help="only with separately recorded authority for exact host-locked archives")
    args = parser.parse_args()
    qualification = Qualification(**vars(args))
    try:
        qualification.execute()
    finally:
        qualification.save()


if __name__ == "__main__":
    main()
