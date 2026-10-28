"""Per-engine storage for the ingest frontier's legacy attributes."""

from dataclasses import dataclass
from typing import Any


@dataclass
class FrontierState:
    """Hold frontier values without changing their transition call sites."""

    cursor: int = 0
    needs_reconcile: bool = False
    compacted_store_id: int = 0
    commit_proof: dict[str, Any] | None = None
    emission_descriptors: dict[str, Any] | None = None

    def invalidate(self, *, drop_proof: bool = False) -> None:
        if drop_proof:
            self.commit_proof = None
        self.cursor = 0
        self.needs_reconcile = True

    def adopt(self, n: int) -> None:
        self.cursor = n

    def reset_progress(self) -> None:
        self.compacted_store_id = 0
        self.cursor = 0
        self.needs_reconcile = False

    def reset_proofs(self) -> None:
        self.commit_proof = None
        self.emission_descriptors = None

    def adopt_trusted(self, n: int) -> None:
        self.cursor = n
        self.needs_reconcile = False

    def restart_cursor(self) -> None:
        self.cursor = 0
        self.compacted_store_id = 0

    def bind_frontier(self, store_id: int) -> None:
        self.compacted_store_id = store_id

    def publish_frontier(self, store_id: int) -> None:
        self.compacted_store_id = store_id

    def filter_by_binding(self, *, proof: bool, descriptors: bool) -> None:
        if proof:
            self.commit_proof = None
        if descriptors:
            self.emission_descriptors = None

    def transfer_to_child(self, target: dict[str, Any], key: str, value: Any) -> None:
        # Keep each write separate: binding checks and reads occur between them.
        target[key] = value

    def mark_end_consumed(self, proof: dict[str, Any]) -> None:
        proof["end_consumed"] = True
