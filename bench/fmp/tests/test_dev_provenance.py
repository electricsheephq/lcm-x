"""Offline provenance regressions for #1687; no model or network calls."""
import json
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from bench.fmp import closeout, dev
from bench.fmp.common import digest_file, sha, write_json
from bench.fmp.tests.test_dev_bot1 import screen as screen

RUN = subprocess.run


def test_packaged_card_contains_only_identity_specs(screen, monkeypatch):
    run, _, _ = screen
    worker_run = dev.subprocess.run
    monkeypatch.setattr(subprocess, "run", RUN)
    monkeypatch.setattr(dev, "subprocess", SimpleNamespace(run=worker_run))
    monkeypatch.setattr(dev, "card_tool", closeout.card_tool)
    args = run()
    card = json.loads((args.output / "RESULT-MANIFEST.json").read_bytes())
    packaged = {args.output / "RESULT-MANIFEST.json", args.output / "RESULT.md"}
    for entry in card["results"]:
        path = args.output / entry["path"]
        packaged.update(p for p in path.rglob("*") if p.is_file()) if path.is_dir() else packaged.add(path)
    for path in packaged:
        text = path.read_text()
        assert str(Path.home()) not in text, path.name
        assert not re.search(r'(?:^|[\s"\'=])/(?!/)[^\s"\']+', text), path.name
    for arm in ("baseline", "candidate"):
        public = json.loads((args.output / "inputs" / f"{arm}-spec.json").read_bytes())
        raw_path = args.output / "local" / f"{arm}-spec.json"
        raw = json.loads(raw_path.read_bytes())
        assert raw_path not in packaged
        assert public["spec_sha256"] == digest_file(raw_path)
        assert public["files"]["panel"] == dict(name=args.panel.name, sha256=digest_file(args.panel))
        assert public["repo"] == args.baseline.name
        assert Path(raw["repo"]).is_absolute()


@pytest.fixture
def worker_setup(tmp_path, monkeypatch):
    product = tmp_path / "product.py"
    product.write_text("before")
    identities = []

    def identity(repo):
        value = sha(product.read_bytes())
        identities.append(value)
        return value

    monkeypatch.setattr(dev, "source_identity", identity)
    monkeypatch.setattr(dev, "load_lcm", lambda repo: None)
    q = SimpleNamespace(qid="synthetic-id", dataset="lme-s", question="synthetic", question_date="")
    monkeypatch.setattr(dev, "load", lambda *a: ([q], {"dataset_sha256": {"lme-s": "a", "locomo": "b"}}, {}))
    monkeypatch.setattr(dev.shutil, "disk_usage", lambda p: SimpleNamespace(free=200 * 1024 ** 3))
    store_dir = tmp_path / "store"
    store_dir.mkdir()
    write_json(store_dir / "build-manifest.json", {})
    (store_dir / "lcm.db").write_bytes(b"synthetic")
    write_json(store_dir / "speakers.json", {})
    store = SimpleNamespace(identity={}, expected=lambda q: {}, directory=lambda q: store_dir)
    monkeypatch.setattr(dev, "Stores", lambda *a: store)
    spec = dict(repo=str(tmp_path), panel="panel", guard="guard", lme="lme", locomo="locomo",
                cache=str(tmp_path), config="off", overrides={}, receipt=str(tmp_path / "receipt.json"))
    return spec, store, product, identities


@pytest.mark.parametrize("prepare", [False, True])
@pytest.mark.parametrize("changed", [False, True])
def test_worker_records_identity_and_rejects_checkout_change(worker_setup, monkeypatch, prepare, changed):
    spec, store, product, identities = worker_setup

    def work(*a):
        if changed:
            product.write_text("after")
        return dict(evidence="synthetic", source_sha256="digest", metadata={})

    store.build = work
    monkeypatch.setattr(dev, "retrieve", work)
    spec["prepare"] = prepare
    if changed:
        with pytest.raises(dev.Incomplete, match="checkout-changed"):
            dev.worker(spec)
    else:
        dev.worker(spec)
    receipt = json.loads(Path(spec["receipt"]).read_bytes())
    assert len(identities) == 2
    assert receipt["product"] == identities[0]
    assert receipt["product_after"] == identities[1]
    assert receipt["status"] == ("INCOMPLETE" if changed else "COMPLETE")
    if changed:
        assert receipt["reason"] == "checkout-changed"


def test_checkout_change_reaches_run_report(screen, worker_setup, monkeypatch):
    spec, store, product, _ = worker_setup
    store.build = lambda q: product.write_text("after")

    def worker_run(cmd, **kw):
        operational = json.loads(Path(cmd[-1]).read_bytes())
        operational.update(spec, receipt=operational["receipt"], prepare=True)
        write_json(Path(cmd[-1]), operational)
        with monkeypatch.context() as worker_patch:
            worker_patch.setattr(sys, "argv", ["dev", "--worker", cmd[-1]])
            return SimpleNamespace(returncode=dev.main())

    monkeypatch.setattr(dev.subprocess, "run", worker_run)
    # The fixture expects success; call execute directly with its standard arguments.
    root = Path(spec["cache"])
    panel = root / "panel.json"
    write_json(panel, {})
    args = dev.parser().parse_args(["--baseline", str(root), "--cache-root", str(root / "cache"),
        "--output", str(root / "failed"), "--panel", str(panel), "--guard", str(panel),
        "--lme-data", str(panel), "--locomo-data", str(panel)])
    assert dev.execute(args) == 2
    report = json.loads((args.output / "report.json").read_bytes())
    assert report["verdict"] == "INCOMPLETE"
    assert report["reason"] == "checkout-changed"
    receipt = json.loads((args.output / "inputs" / "baseline-retrieval.json").read_bytes())
    assert receipt["product"] != receipt["product_after"]
    assert receipt["status"] == "INCOMPLETE" and receipt["reason"] == "checkout-changed"
