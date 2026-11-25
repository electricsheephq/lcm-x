"""Pin cloud reader constants to the documented and runtime contract."""
from pathlib import Path
import re

from hermes_lcm import cloud_copy as cp, claude_code_store as cc, externalize, ingest_content, db_bootstrap
from scripts.store_contract_gate import contract


def test_25_compat():
    text = (Path(__file__).resolve().parents[2] / 'docs/store-contract.md').read_text()
    assert (cp.CONTRACT_VERSION, cp.CONTRACT_DIGEST) == contract(text)
    tables = {}
    for line in text.splitlines():
        match = re.match(r'\| `([a-z_]+)` \| (.*?) \|', line)
        if match and ': ' in match[2]:
            tables[match[1]] = re.findall(r'(\w+): [A-Z]+;', match[2])
    assert cp.TABLE_COLUMNS == tables and len(tables) == 15
    assert cp.EXTERNALIZED_REF_RE.pattern == externalize._EXTERNALIZED_REF_RE.pattern
    assert cp.INGEST_PLACEHOLDER_RE.pattern == ingest_content._INGEST_PLACEHOLDER_RE.pattern
    assert cp.GENERATED_PAYLOAD_REF_RE.pattern == externalize._GENERATED_PAYLOAD_REF_RE.pattern
    assert cp.FORGOTTEN_PREFIX == cc.FORGOTTEN_PREFIX
    assert cp.OWNER_KEY == cc.OWNER_KEY
    assert cp.PAYLOAD_FOLDER == externalize.DEFAULT_LARGE_OUTPUT_DIRNAME
    assert cp.SCHEMA_VERSION == db_bootstrap.SCHEMA_VERSION == 5
