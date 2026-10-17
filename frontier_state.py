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
