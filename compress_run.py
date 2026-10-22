"""Ownership of the store-ID map used by compaction and its temporary readers."""

from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass, field


@dataclass
class CompressRun:
    """The active map; pass-level replacements retain their existing lifetime."""

    store_ids_by_message_id: dict[int, int] = field(default_factory=dict)

    def preserve_store_ids(self) -> AbstractContextManager[None]:
        """Capture now, then restore around the caller's existing protected body.

        Map construction stays before entry, just as it preceded the old try.
        A construction failure therefore never enters the restoration boundary.
        """
        previous = self.store_ids_by_message_id

        @contextmanager
        def scope() -> Iterator[None]:
            try:
                yield
            finally:
                self.store_ids_by_message_id = previous

        return scope()
