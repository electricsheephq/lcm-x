from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from access_context import (
    AccessContextV1,
    Decision,
    DenialReason,
    ResolutionMode,
    VALIDATION_ORDER,
    ValidationStage,
    derive_child,
    is_subset_of,
    resolve_mode,
    validate,
)
from access_context.denials import PUBLIC_DENIAL_PROJECTION, project_public
from access_context.fixtures import fixture_paths, load_context, load_fixture
from access_context.protocols import HostContextCarrier, LcmAuthorizationConsumer


NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _context(path: Path) -> AccessContextV1:
    context = load_context(path)
    assert context is not None
    return context


def test_validation_order_is_explicit_and_stable() -> None:
    assert VALIDATION_ORDER == (
        ValidationStage.CONTEXT_PRESENT,
        ValidationStage.CONTEXT_WELL_FORMED,
        ValidationStage.REVISION_SUPPORTED,
        ValidationStage.NOT_EXPIRED,
        ValidationStage.NOT_REVOKED,
        ValidationStage.OWNERSHIP_CURRENT,
        ValidationStage.LEASE_CURRENT,
        ValidationStage.SCOPE_PERMITTED,
        ValidationStage.TARGET_RESOLUTION,
    )


def test_positive_contexts_validate() -> None:
    paths = fixture_paths("positive")
    assert len(paths) >= 5
    for path in paths:
        context = _context(path)
        assert validate(context, required_scope="read", now=NOW).allowed, path


def _expected_decision(path: Path):
    payload = load_fixture(path)
    expected = payload["expected"]
    kwargs = dict(expected.get("validate_kwargs", {}))
    now = datetime.fromisoformat(expected.get("now", "2026-01-01T00:00:00+00:00").replace("Z", "+00:00"))
    required_scope = expected.get("required_scope")
    if "target_allowed" in expected:
        kwargs["target_allowed"] = expected["target_allowed"]
    if "requested_collections" in expected:
        kwargs["requested_collections"] = expected["requested_collections"]
    return validate(_context(path) if payload["context"] is not None else None, required_scope=required_scope, now=now, **kwargs)


def test_negative_fixtures_are_parametrized_and_cover_every_denial() -> None:
    paths = fixture_paths("negative")
    assert len(paths) >= len(tuple(DenialReason))
    discovered = set()
    for path in paths:
        decision = _expected_decision(path)
        expected_reason = load_fixture(path)["expected"]["denial_reason"]
        discovered.add(expected_reason)
        assert not decision.allowed, path
        assert decision.denial_reason.value == expected_reason, path
    assert discovered == {reason.value for reason in DenialReason}


def test_expiry_wins_scope_tie_break() -> None:
    path = Path("tests/fixtures/access_context_v1/negative/expired-and-out-of-scope.json")
    decision = _expected_decision(path)
    assert decision.denial_reason is DenialReason.CONTEXT_EXPIRED


def test_delegation_vectors_prove_subset_and_reject_each_widening() -> None:
    paths = fixture_paths("delegation")
    assert len(paths) >= 10
    widening_fields = set()
    for path in paths:
        payload = load_fixture(path)
        if "candidate" not in payload:
            continue
        parent = AccessContextV1.from_payload(payload["context"])
        candidate = AccessContextV1.from_payload(payload["candidate"])
        expected = payload["expected"]
        assert is_subset_of(candidate, parent) is expected["subset"], path
        if not expected["subset"]:
            widening_fields.add(expected["widen_field"])
    assert {"operations", "collections", "audience", "profile_binding", "session_binding", "expiry", "policy_revision", "membership_revision", "revocation_epoch", "ownership_generation", "lease_generation"} <= widening_fields


def test_three_deep_redelegation_preserves_chain_and_narrowing() -> None:
    root = _context(Path("tests/fixtures/access_context_v1/delegation/redelegation-chain-3-deep.json"))
    one = derive_child(root, operations=["read"], collections=["collection-a"], audience=["profile-a"], expires_at="2026-09-01T00:00:00Z")
    two = derive_child(one, operations=["read"], collections=["collection-a"], audience=["profile-a"], expires_at="2026-08-01T00:00:00Z")
    three = derive_child(two, operations=["read"], collections=["collection-a"], audience=["profile-a"], expires_at="2026-07-01T00:00:00Z")
    assert three.delegation_chain == (root.context_id, one.context_id, two.context_id)
    assert len(three.delegation_chain) == 3
    assert root.narrowing <= three.narrowing
    assert is_subset_of(one, root)
    assert is_subset_of(two, one)
    assert is_subset_of(three, two)


def test_revocation_vectors_invalidate_the_next_use() -> None:
    paths = fixture_paths("revocation")
    assert len(paths) >= 4
    for path in paths:
        payload = load_fixture(path)
        decision = _expected_decision(path)
        assert decision.denial_reason.value == payload["expected"]["denial_reason"], path


def test_absent_carrier_compatibility_matrix() -> None:
    context = _context(Path("tests/fixtures/access_context_v1/positive/human.json"))
    assert resolve_mode(None, False) is ResolutionMode.STANDARD_UNMANAGED
    assert resolve_mode(None, True) is ResolutionMode.FAIL_CLOSED
    assert resolve_mode(context, False) is ResolutionMode.STANDARD_UNMANAGED
    assert resolve_mode(context, True) is ResolutionMode.ENFORCING
    assert validate(None, now=NOW).denial_reason is DenialReason.CONTEXT_MISSING


def test_public_denial_projection_is_total_and_content_free() -> None:
    assert set(PUBLIC_DENIAL_PROJECTION) == set(DenialReason)
    for reason in DenialReason:
        internal = Decision.deny(reason, context_id="ctx", query_text="must not escape")
        public = project_public(internal)
        assert public.denial_reason is PUBLIC_DENIAL_PROJECTION[reason]
        assert "query_text" not in internal.detail


def test_two_principal_replay_vectors_fail_closed() -> None:
    paths = [path for path in fixture_paths("negative") if "replay-" in path.name]
    assert len(paths) >= 6
    for path in paths:
        decision = _expected_decision(path)
        assert not decision.allowed, path
        assert decision.denial_reason in {
            DenialReason.CONTEXT_INVALID,
            DenialReason.OWNERSHIP_CHANGED,
            DenialReason.LEASE_STALE,
        }


def test_concurrent_contexts_do_not_cross_contaminate_threads_or_tasks() -> None:
    human = _context(Path("tests/fixtures/access_context_v1/positive/human.json"))
    agent = _context(Path("tests/fixtures/access_context_v1/positive/agent.json"))

    def check(context: AccessContextV1):
        return validate(context, required_scope="read", now=NOW).detail["context_id"]

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert set(pool.map(check, (human, agent))) == {human.context_id, agent.context_id}

    async def run_pair():
        return await asyncio.gather(asyncio.to_thread(check, human), asyncio.to_thread(check, agent))

    # Python 3.9 has no asyncio.to_thread; executor-backed coroutines preserve
    # the same task-isolation assertion without introducing context globals.
    async def run_pair_39():
        loop = asyncio.get_running_loop()
        return await asyncio.gather(loop.run_in_executor(None, check, human), loop.run_in_executor(None, check, agent))

    assert asyncio.run(run_pair_39()) == [human.context_id, agent.context_id]


def test_validation_module_has_no_mutable_context_store() -> None:
    import access_context.validation as validation_module

    assert isinstance(validation_module.VALIDATION_ORDER, tuple)
    assert not any(
        isinstance(value, (dict, list, set))
        for name, value in vars(validation_module).items()
        if not name.startswith("__") and name not in {"annotations"}
    )


class RecordingConsumer:
    """Reference fake used only to freeze authorization/disclosure ordering."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def authorize(self, context, operation, target_scope):
        self.calls.append("authorize")
        return Decision.allow()

    def select_collection(self, target_scope):
        self.calls.append("select_collection")
        return "collection"

    def count_candidates(self, candidates):
        self.calls.append("count_candidates")
        return len(candidates)

    def rank_candidates(self, candidates):
        self.calls.append("rank_candidates")
        return candidates

    def hydrate_targets(self, targets):
        self.calls.append("hydrate_targets")
        return targets

    def issue_handle(self, target):
        self.calls.append("issue_handle")
        return "handle"


class RecordingCarrier:
    def __init__(self, context):
        self.context = context

    def get_access_context(self):
        return self.context


def test_protocol_order_authorize_precedes_every_disclosure_primitive() -> None:
    context = _context(Path("tests/fixtures/access_context_v1/positive/human.json"))
    consumer = RecordingConsumer()
    carrier = RecordingCarrier(context)
    assert isinstance(carrier, HostContextCarrier)
    assert isinstance(consumer, LcmAuthorizationConsumer)
    consumer.authorize(carrier.get_access_context(), "read", {"collection": "collection-main"})
    consumer.select_collection({})
    consumer.count_candidates([1])
    consumer.rank_candidates([1])
    consumer.hydrate_targets([1])
    consumer.issue_handle(1)
    assert consumer.calls == [
        "authorize",
        "select_collection",
        "count_candidates",
        "rank_candidates",
        "hydrate_targets",
        "issue_handle",
    ]
