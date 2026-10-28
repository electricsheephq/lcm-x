"""Typed verifier findings and the measured embedding compatibility boundary.

DDL remains owned by db_bootstrap; registry tuples are checked against it by A5.
"""

from dataclasses import dataclass

FINDING_KINDS = (
    "missing object",
    "malformed table",
    "malformed constraints",
    "malformed index",
)


@dataclass(frozen=True)
class SchemaFinding:
    kind: str
    name: str

    def __post_init__(self) -> None:
        if self.kind not in FINDING_KINDS:
            raise ValueError(f"unknown schema finding kind: {self.kind}")

    def __str__(self) -> str:
        return f"{self.kind}: {self.name}"


@dataclass(frozen=True)
class SchemaFamily:
    name: str
    prefix: str
    marker: str
    tables: tuple[str, ...]
    indexes: tuple[str, ...]
    oldest_reader: str
    removal_version: str | None


COMPAT_REGISTRY: dict[str, SchemaFamily] = {
    "embedding": SchemaFamily(
        name="embedding",
        prefix="lcm_embedding",
        marker="embeddings_v1",
        tables=(
            "lcm_embedding_profile",
            "lcm_embedding_meta",
            "lcm_embedding_vectors",
            "lcm_embedding_binary",
        ),
        indexes=(
            "idx_lcm_embedding_profile_model",
            "idx_lcm_embedding_meta_identity_embedded_at",
        ),
        oldest_reader="v0.20.0",
        removal_version=None,
    ),
}

for _name, _prefix, _tables in (
    ("entity", "lcm_entit", ("lcm_entities", "lcm_entity_aliases", "lcm_entity_keys")),
    ("memory", "lcm_memory_", ("lcm_memory_notes", "lcm_memory_events", "lcm_memory_host_file_revisions")),
):
    COMPAT_REGISTRY[_name] = SchemaFamily(
        name=_name, prefix=_prefix, marker="entity_memory_v1", tables=_tables,
        indexes=(("idx_lcm_entity_platform_id", "idx_lcm_entity_key_active") if _name == "entity"
                 else ("idx_lcm_memory_host_file_rev_content",)),
        oldest_reader="v0.25.0-rc1", removal_version=None,
    )
