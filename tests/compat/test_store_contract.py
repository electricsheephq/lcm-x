"""Fingerprint the consumer surface on an isolated, freshly initialized store."""
import hashlib
import inspect
import json
from pathlib import Path

from hermes_lcm import claude_code_store, db_bootstrap, mcp_reader, store
from hermes_lcm.dag import SummaryDAG
from scripts.store_contract_gate import DOC, contract

TABLES = sorted("messages summary_nodes metadata lcm_migration_state lcm_lifecycle_state host_uid_bindings "
                "message_relations session_lineage ledger_meta occurrence_bindings ledger_emissions "
                "stamp_observations ingest_views lineage_marks capture_events".split())
CALLABLES = [store.MessageStore, SummaryDAG, db_bootstrap.mark_migration_step_complete, store.register_session_purge,
             mcp_reader.StoreReader, mcp_reader.handle,
             *(getattr(store.MessageStore, name) for name in (
                 "append_batch", "ingest_atomic", "atomic", "add_host_uid_bindings", "add_message_relations",
                 "record_session_lineage", "host_uid_uids_of_stores", "read_metadata_json", "update_metadata_json", "close")),
             store.MessageStore.connection.fget,
             *(getattr(mcp_reader.StoreReader, name) for name in ("prepare", "envelope", "call"))]


def surface(directory):
    writer = store.MessageStore(directory / "contract.db", hermes_home=str(directory))
    try:
        SummaryDAG(writer.db_path).close()
        writer._ensure_identity_anchor_schema()
        writer._ensure_host_uid_schema()
        writer.ensure_ingest_ledger()
        writer.record_session_lineage("child", "parent", source="contract-test")
        writer.update_metadata_json("capture:owner", lambda _: {"host": "claude-code", "format": 1})
        claude_code_store.ensure_capture_events(writer)
        tables = {name: {"columns": writer.connection.execute(f'PRAGMA table_info("{name}")').fetchall(),
                         "indexes": sorted(row[1] for row in writer.connection.execute(f'PRAGMA index_list("{name}")'))}
                  for name in TABLES}
        assert all(table["columns"] for table in tables.values()), "contracted table missing"
        signatures = {}
        for method in CALLABLES:
            signature = inspect.signature(method)
            signatures[method.__module__.removeprefix("hermes_lcm.") + "." + method.__qualname__] = str(signature.replace(
                parameters=[p.replace(annotation=inspect.Parameter.empty) for p in signature.parameters.values()],
                return_annotation=inspect.Signature.empty))
        return {"tables": tables, "callables": dict(sorted(signatures.items())), "SCHEMA_VERSION": db_bootstrap.SCHEMA_VERSION}
    finally:
        writer.close()


def test_store_surface_digest(tmp_path):
    payload = json.dumps(surface(tmp_path), sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    digest = "sha256:" + hashlib.sha256(payload).hexdigest()
    _, documented = contract((Path(__file__).resolve().parents[2] / DOC).read_text(encoding="utf-8"))
    assert digest == documented, "store surface digest differs from docs/store-contract.md"


def test_store_contract_changelog():
    contract((Path(__file__).resolve().parents[2] / DOC).read_text(encoding="utf-8"))


def _rendered_rows(found):
    rows = set()
    for name, table in found["tables"].items():
        columns = "; ".join(f"{c[1]}: {c[2]}; {'NOT NULL' if c[3] else 'nullable'}; "
                            f"{'no default' if c[4] is None else c[4]}; pk={c[5]}" for c in table["columns"])
        rows.add(f"| `{name}` | {columns} | {', '.join(table['indexes'])} |")
    rows.update(f"| `{name}` | `{signature}` |" for name, signature in found["callables"].items())
    return rows


def test_store_contract_doc_tables_match_surface(tmp_path):
    """The doc's generated tables state exactly the hashed surface: no row missing, none stale."""
    sections, heading = {}, ""
    for line in (Path(__file__).resolve().parents[2] / DOC).read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            heading = line
        elif line.startswith("| `"):
            sections.setdefault(heading, set()).add(line)
    documented = sections.get("## What a consumer may read", set()) | sections.get("## How a consumer may write", set())
    rendered = _rendered_rows(surface(tmp_path))
    assert sorted(rendered - documented) == [], "surface rows missing from docs/store-contract.md"
    assert sorted(documented - rendered) == [], "docs/store-contract.md lists rows the surface no longer has"
