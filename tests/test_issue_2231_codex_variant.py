"""Host-validated long-context variants preserve conservative base caps."""

import sys
from types import ModuleType

import pytest

from hermes_lcm.compaction_policy import effective_context_length


@pytest.mark.parametrize("helper", ["validated", "rejected", "missing", "module-missing"])
@pytest.mark.parametrize("model", ["gpt-5.6-sol", "gpt-5.6-sol-900k", "openai/gpt-5.6-sol-900k"])
def test_codex_variant_requires_host_validation(monkeypatch, helper, model):
    metadata = ModuleType("agent.model_metadata")
    if helper in {"validated", "rejected"}:
        metadata.is_codex_context_variant = lambda slug: helper == "validated" and slug == "gpt-5.6-sol-900k"
    monkeypatch.setitem(sys.modules, "agent.model_metadata", None if helper == "module-missing" else metadata)
    if helper == "validated" and model.endswith("-900k"):
        assert effective_context_length(900_000, model, "openai-codex") == (900_000, None, "")
    else:
        assert effective_context_length(900_000, model, "openai-codex") == (
            272_000, 272_000, "codex_oauth_context_cap")
    assert effective_context_length(900_000, model, "openai") == (900_000, None, "")
