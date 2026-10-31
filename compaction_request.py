"""One-shot preflight instructions consumed by the next foreground compaction."""

from dataclasses import dataclass


@dataclass
class CompactionRequest:
    """Pending handoffs; each entry point retains its existing reset policy."""

    # Adopt already-durable replay cleanup during boundary cooldown without summary work.
    cleanup_only_due_to_boundary_cooldown: bool = False
    # #651: preflight requested maintenance below the host threshold, so only clean up.
    below_threshold_cleanup_only: bool = False
    # #677: any preflight request; compress() cleans up only below the host's count.
    automatic_request: bool = False
    # #618: a hold is active at the survival ceiling, so this pass only fits.
    hold_fit_only_requested: bool = False
