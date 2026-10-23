"""Scoped entity-memory primitives; schema creation belongs to db_bootstrap."""

from contextlib import contextmanager
from pathlib import Path
import json
import sqlite3
import threading
import time
import unicodedata
from urllib.parse import quote
import uuid

from .db_bootstrap import (ENTITY_MIGRATION_STEP, configure_connection, ensure_entity_tables,
                           mark_migration_step_complete, refuse_schema_version_too_new,
                           run_versioned_migrations, verify_entity_schema)
from .sqlite_session import ConnectionPolicy, open_connection


class EntityStoreUnavailableError(RuntimeError):
    """The entity schema is absent or incompatible with this reader."""


def normalize_alias(value: str) -> str:
    value = " ".join(unicodedata.normalize("NFKC", value).casefold().split())
    while value and unicodedata.category(value[0]).startswith("P"):
        value = value[1:].lstrip()
    while value and unicodedata.category(value[-1]).startswith("P"):
        value = value[:-1].rstrip()
    return value


def normalize_platform_id(platform: str, identifier: str) -> str:
    return f"{platform.lower()}:{identifier}"


class EntityStore:
    """One profile's source records, with atomic mutation/event pairs.

    A callable profile_key follows an engine's bound profile identity on
    configured-database rebinds. Access stamps are discovered on open, after
    the core store's policy setup; policy/schema transitions require reopening.
    """

    def __init__(self, db_path, *, profile_key, read_only=False, id_factory=None, now=None):
        self.db_path, self.read_only = Path(db_path), read_only
        self._profile_key, self._ids, self._now = profile_key, id_factory or (lambda: uuid.uuid4().hex), now or time.time
        self._lock, self._depth = threading.RLock(), 0
        database = f"file:{quote(str(self.db_path), safe='/')}?mode=ro" if read_only else str(self.db_path)
        if not read_only:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = open_connection(ConnectionPolicy(
            database=database, connect_kwargs=(("uri", read_only), ("timeout", 5.0),
                ("check_same_thread", False), ("isolation_level", None)),
            before_row_factory=(("PRAGMA query_only=ON", "PRAGMA busy_timeout=30000") if read_only
                                else (refuse_schema_version_too_new, configure_connection)),
            row_factory=sqlite3.Row, on_open=None, on_setup_error=None))
        try:
            self._init_db()
            self._stamp = "m.access_scope" if any(r[1] == "access_scope" for r in
                self._conn.execute("PRAGMA table_info(messages)")) else "NULL"
        except BaseException:
            self._conn.close()
            raise

    @property
    def scope_key(self):
        return self._profile_key() if callable(self._profile_key) else self._profile_key

    @property
    def connection(self):
        return self._conn

    def _init_db(self):
        if not self.read_only:
            run_versioned_migrations(self._conn)
            ensure_entity_tables(self._conn)
        findings = verify_entity_schema(self._conn)
        if findings and not self.read_only:
            ensure_entity_tables(self._conn)
            findings = verify_entity_schema(self._conn)
        if findings:
            raise EntityStoreUnavailableError("; ".join(findings))
        if not self.read_only and not self._conn.execute(
                "SELECT 1 FROM lcm_migration_state WHERE step_name=?", (ENTITY_MIGRATION_STEP,)).fetchone():
            mark_migration_step_complete(self._conn, ENTITY_MIGRATION_STEP)

    def close(self):
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    @contextmanager
    def transaction(self):
        with self._lock:
            if self.read_only:
                raise PermissionError("read-only EntityStore")
            depth = self._depth
            self._conn.execute("BEGIN IMMEDIATE" if depth == 0 else f"SAVEPOINT entity_{depth}")
            self._depth += 1
            try:
                yield self
                self._conn.execute("COMMIT" if depth == 0 else f"RELEASE entity_{depth}")
            except BaseException:
                self._conn.execute("ROLLBACK" if depth == 0 else f"ROLLBACK TO entity_{depth}")
                if depth:
                    self._conn.execute(f"RELEASE entity_{depth}")
                raise
            finally:
                self._depth -= 1

    def _insert(self, table, values):
        values = dict(values, scope_key=self.scope_key)
        return self._conn.execute(f"INSERT INTO {table} ({','.join(values)}) VALUES ({','.join('?' for _ in values)})",
                                  tuple(values.values())).lastrowid

    def _mutate(self, action, op, payload, *, log_event=True, actor="agent", now=None):
        with self._lock:
            if not log_event and not self._depth:
                raise RuntimeError("log_event=False requires transaction()")
            with self.transaction():
                result = action()
                payload["row_id"] = result
                if log_event:
                    self.append_event(op, payload, actor=actor, now=now)
                return result

    def _id(self, prefix, factory):
        return prefix + (factory or self._ids)()

    def create_entity(self, display_name, kind="person", *, id_factory=None, now=None, log_event: bool = True, **event):
        key, stamp = self._id("ent_", id_factory), (now or self._now)()
        row = dict(entity_id=key, kind=kind, display_name=display_name, status="active", merged_into=None, created_at=stamp)
        self._mutate(lambda: self._insert("lcm_entities", row), "create_entity", row, now=lambda: stamp, log_event=log_event, **event)
        return key

    def add_alias(self, entity_id, alias_display, alias_kind="name", *, alias_norm=None,
                  owner_entity_id=None, source_store_ids=(), now=None, log_event: bool = True, **event):
        stamp = (now or self._now)()
        if alias_kind == "platform_id" and alias_norm is None:
            raise ValueError("platform_id requires normalize_platform_id output")
        row = dict(entity_id=entity_id, alias_display=alias_display, alias_kind=alias_kind,
                   alias_norm=alias_norm if alias_norm is not None else normalize_alias(alias_display),
                   owner_entity_id=owner_entity_id, source_store_ids=json.dumps(list(source_store_ids)), created_at=stamp)
        return self._mutate(lambda: self._insert("lcm_entity_aliases", row), "add_alias", row, now=lambda: stamp, log_event=log_event, **event)

    def _project(self, table, key_column, key, changes, op, **event):
        def action():
            before = self._conn.execute(f"SELECT * FROM {table} WHERE {key_column}=? AND scope_key=?",
                                        (key, self.scope_key)).fetchone()
            if before is None:
                raise KeyError(key)
            payload.update(before=dict(before), after=changes)
            self._conn.execute(f"UPDATE {table} SET {','.join(k+'=?' for k in changes)} "
                               f"WHERE {key_column}=? AND scope_key=?", (*changes.values(), key, self.scope_key))
        payload = {key_column: key}
        self._mutate(action, op, payload, **event)

    def retire_alias(self, alias_id, *, now=None, log_event: bool = True, **event):
        stamp = (now or self._now)()
        self._project("lcm_entity_aliases", "alias_id", alias_id, {"retired_at": stamp}, "retire_alias", now=lambda: stamp, log_event=log_event, **event)

    def restore_alias(self, alias_id, *, log_event: bool = True, **event):
        self._project("lcm_entity_aliases", "alias_id", alias_id, {"retired_at": None}, "restore_alias", log_event=log_event, **event)

    def set_entity_status(self, entity_id, status, merged_into=None, *, log_event: bool = True, **event):
        self._project("lcm_entities", "entity_id", entity_id, dict(status=status, merged_into=merged_into), "set_entity_status", log_event=log_event, **event)

    def add_note(self, text, *, kind="fact", entity_id=None, about_entity_id=None, asserted_by_entity_id=None,
                 cites=(), cites_file_revisions=(), author="agent", superseded_by=None,
                 retracts_note_id=None, retracts_store_ids=(), id_factory=None, now=None, log_event: bool = True, **event):
        key, stamp = self._id("note_", id_factory), (now or self._now)()
        row = dict(note_id=key, text=text, kind=kind, entity_id=entity_id, about_entity_id=about_entity_id,
                   asserted_by_entity_id=asserted_by_entity_id, cites=json.dumps(list(cites)),
                   cites_file_revisions=json.dumps(list(cites_file_revisions)), author=author,
                   superseded_by=superseded_by, retracts_note_id=retracts_note_id,
                   retracts_store_ids=json.dumps(list(retracts_store_ids)), created_at=stamp)
        self._mutate(lambda: self._insert("lcm_memory_notes", row), "add_note", row, now=lambda: stamp, log_event=log_event, **event)
        return key

    def set_note_superseded(self, note_id, superseded_by_or_None, *, log_event: bool = True, **event):
        self._project("lcm_memory_notes", "note_id", note_id, {"superseded_by": superseded_by_or_None}, "supersede_note", log_event=log_event, **event)

    def append_event(self, op, payload, *, actor="agent", target_event_id=None, now=None, log_event: bool = True):
        with self._lock:
            if not log_event and not self._depth:
                raise RuntimeError("log_event=False requires transaction()")
            if not log_event:
                return None
            with self.transaction():
                return self._insert("lcm_memory_events", dict(op=op, payload_json=json.dumps(payload), actor=actor,
                                    target_event_id=target_event_id, created_at=(now or self._now)()))

    def insert_host_file_revision(self, path, revision_sha, content_hash, *, id_factory=None, now=None, log_event: bool = True, **event):
        key, stamp = self._id("rev_", id_factory), (now or self._now)()
        row = dict(revision_id=key, path=path, revision_sha=revision_sha, content_hash=content_hash,
                   profile_key=self.scope_key, captured_at=stamp)
        with self._lock:
            if not log_event and not self._depth:
                raise RuntimeError("log_event=False requires transaction()")
            with self.transaction():
                prior = self._conn.execute("SELECT revision_id FROM lcm_memory_host_file_revisions "
                    "WHERE scope_key=? AND profile_key=? AND path=? AND content_hash=?",
                    (self.scope_key, self.scope_key, path, content_hash)).fetchone()
                if prior:
                    return prior[0]
                return self._mutate(lambda: (self._insert("lcm_memory_host_file_revisions", row), key)[1],
                                    "host_file_revision", row, now=lambda: stamp, log_event=log_event, **event)

    def _visible(self, sources, revisions="'[]'", *, empty_owner=False):
        return f"""NOT EXISTS (SELECT 1 FROM json_each({sources}) c LEFT JOIN messages m ON m.store_id=c.value
            WHERE m.store_id IS NULL OR (:principal IS NOT NULL AND {self._stamp} IS NOT NULL AND {self._stamp}!=:principal))
            AND NOT EXISTS (SELECT 1 FROM json_each({revisions}) c LEFT JOIN lcm_memory_host_file_revisions r
                ON r.revision_id=c.value AND r.scope_key=:scope AND r.profile_key=:scope
                WHERE r.revision_id IS NULL OR (:principal IS NOT NULL AND NOT :owner))
            AND (:principal IS NULL OR json_array_length({sources})+json_array_length({revisions})>0
                 OR ({int(empty_owner)} AND :owner))"""

    def _read(self, table, where="1", params=None, *, access_scope=None, principal_is_owner=False, order="rowid"):
        if access_scope is not None and not isinstance(principal_is_owner, bool):
            raise TypeError("principal_is_owner must be a bool in principal mode")
        visible = "1"
        if table == "lcm_entity_aliases" or table == "lcm_entity_keys":
            visible = self._visible("t.source_store_ids", empty_owner=table == "lcm_entity_aliases")
        elif table == "lcm_memory_notes":
            visible = self._visible("t.cites", "t.cites_file_revisions")
        elif table == "lcm_memory_host_file_revisions":
            visible = "t.profile_key=:scope AND (:principal IS NULL OR :owner)"
        elif table == "lcm_entities":
            visible = ("(:principal IS NULL OR EXISTS (SELECT 1 FROM lcm_entity_aliases a WHERE a.entity_id=t.entity_id "
                       "AND a.scope_key=:scope AND " + self._visible("a.source_store_ids", empty_owner=True) + ") OR EXISTS "
                       "(SELECT 1 FROM lcm_memory_notes n WHERE n.entity_id=t.entity_id AND n.scope_key=:scope AND "
                       + self._visible("n.cites", "n.cites_file_revisions") + "))")
        with self._lock:
            return [dict(r) for r in self._conn.execute(f"SELECT t.* FROM {table} t WHERE t.scope_key=:scope "
                f"AND (:principal IS NULL OR :principal!='') AND ({where}) AND ({visible}) ORDER BY {order}",
                dict(params or {}, scope=self.scope_key, principal=access_scope, owner=bool(principal_is_owner)))]

    def get_entity(self, entity_id, **access):
        return next(iter(self._read("lcm_entities", "t.entity_id=:id", {"id": entity_id}, **access)), None)

    def get_note(self, note_id, **access):
        return next(iter(self._read("lcm_memory_notes", "t.note_id=:id", {"id": note_id}, **access)), None)

    def get_alias(self, alias_id, **access):
        return next(iter(self._read("lcm_entity_aliases", "t.alias_id=:id", {"id": alias_id}, **access)), None)

    def get_host_file_revision(self, revision_id, **access):
        return next(iter(self._read("lcm_memory_host_file_revisions", "t.revision_id=:id", {"id": revision_id}, **access)), None)

    def aliases_by_norm(self, alias_norm, **access):
        return self._read("lcm_entity_aliases", "t.alias_norm=:norm AND t.retired_at IS NULL", {"norm": alias_norm}, **access)

    def aliases_for_entity(self, entity_id, **access):
        return self._read("lcm_entity_aliases", "t.entity_id=:id AND t.retired_at IS NULL", {"id": entity_id}, **access)

    def notes_for_entity(self, entity_id, **access):
        return self._read("lcm_memory_notes", "t.entity_id=:id", {"id": entity_id}, **access)

    def keys_for_entity(self, entity_id, **access):
        return self._read("lcm_entity_keys", "t.entity_id=:id AND t.retired_at IS NULL", {"id": entity_id}, **access)

    def events_after(self, event_id=0, **access):
        """Return scope-isolated events for trusted replay only."""
        if 'access_scope' in access or 'principal_is_owner' in access:
            raise ValueError("events are trusted replay only")
        return self._read("lcm_memory_events", "t.event_id>:id", {"id": event_id}, order="event_id", **access)
