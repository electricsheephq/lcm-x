"""Default-off entity operations. Source writes belong to EntityStore."""

from functools import wraps
from contextlib import contextmanager
import json
import sqlite3

from .entity_store import normalize_alias, normalize_platform_id


class EntityOpRefused(ValueError):
    """A scoped, non-mutating refusal with machine-readable detail."""

    def __init__(self, reason, **detail):
        self.reason = reason
        self.__dict__.update(detail)
        super().__init__(reason)


class EntityIntegrityError(RuntimeError):
    """A canonical chain is corrupt."""


FAMILIES = {"entities": ("lcm_entities", "entity_id"),
            "aliases": ("lcm_entity_aliases", "alias_id"),
            "notes": ("lcm_memory_notes", "note_id"),
            "keys": ("lcm_entity_keys", "key_id")}


def _platform_id(pid):
    parts = pid.split(":", 1)
    if len(parts) != 2 or not all(parts):
        raise EntityOpRefused("invalid_platform_id")
    return normalize_platform_id(*parts)


def _records(store, family):
    table, _ = FAMILIES[family]
    return [dict(r) for r in store.connection.execute(
        f"SELECT * FROM {table} WHERE scope_key=?", (store.scope_key,))]


def _history(store):
    events = [dict(e, payload=json.loads(e["payload_json"])) for e in store.events_after()]
    undone = {e["target_event_id"] for e in events if e["op"] == "undo"}
    void = {f: set() for f in FAMILIES}
    kinds = {}
    for event in events:
        p = event["payload"]
        if event["event_id"] in undone:
            for family, ids in p.get("created", {}).items():
                void[family].update(ids)
        elif event["op"] == "change_kind":
            kinds[p["kind"]["entity_id"]] = p["kind"]["after"]
    return events, undone, void, kinds


def _canonical(entities, key):
    if key not in entities:
        raise EntityOpRefused("not_found")
    for _ in range(len(entities)):
        row = entities.get(key)
        if row is None:
            raise EntityOpRefused("not_found")
        if row["merged_into"] is None:
            return key
        key = row["merged_into"]
    raise EntityIntegrityError("canonical hop limit exceeded")


def _operation(method):
    @wraps(method)
    def run(self, *args, **kwargs):
        with self.store.transaction():
            p = {"created": {f: [] for f in FAMILIES}, "changes": [],
                 "touches": {f: [] for f in FAMILIES}}
            try:
                result = method(self, p, *args, **kwargs)
            except sqlite3.IntegrityError as error:
                if method.__name__ != "override_key":
                    raise
                raise EntityOpRefused("key_held") from error
            event = self.store.append_event(method.__name__, p)
            return event if result is None else result
    return run


def _inverse(ops, payload):
    # Retire copies before restoring originals, including platform ids.
    for alias in payload["created"]["aliases"]:
        ops._project(payload=ops._undo_payload, family="aliases", key=alias, retired_at=True)
    for key in payload["created"].get("keys", []):
        ops._project(ops._undo_payload, "keys", key, retired_at=True)
    changes = list(reversed(payload["changes"]))
    for change in sorted(changes, key=lambda c: c["family"] != "aliases"):
        ops._project(ops._undo_payload, change["family"], change["id"], **change["before"])


# W2 may register an inverse(ops, payload) using the same payload/touch contract.
INVERSE_REGISTRY = dict.fromkeys(
    ("create_entity", "add_alias", "retire_alias", "change_kind", "merge", "split", "add_note", "override_key"), _inverse)


class EntityOps:
    """One profile's transactional operations; ids returned by creators are opaque."""

    def __init__(self, store):
        self.store = store

    def _get(self, family, key):
        getter = {"entities": self.store.get_entity, "aliases": self.store.get_alias,
                  "notes": self.store.get_note, "keys": lambda key: next((r for r in
                      _records(self.store, "keys") if r["key_id"] == key), None)}[family]
        row = getter(key)
        if row is None or key in _history(self.store)[2][family]:
            raise EntityOpRefused("not_found")
        return row

    def _touch(self, p, family, *keys):
        for key in keys:
            if key is not None and key not in p["touches"][family]:
                p["touches"][family].append(key)
                if family == "aliases":
                    row = self.store.get_alias(key)
                    if row["alias_kind"] == "platform_id" and row["alias_norm"] not in p["touches"][family]:
                        p["touches"][family].append(row["alias_norm"])

    def _created(self, p, family, key):
        p["created"][family].append(key)
        self._touch(p, family, key)
        return key

    def _project(self, payload, family, key, **after):
        row = self._get(family, key)
        before = {field: row[field] for field in after}
        if family == "entities":
            self.store.set_entity_status(key, **after, log_event=False)
        elif family == "notes":
            self.store.set_note_superseded(key, after["superseded_by"], log_event=False)
        elif family == "keys":
            if after["retired_at"] is None:
                self.store.restore_key(key)
            else:
                self.store.retire_key(key, now=None if after["retired_at"] is True else lambda: after["retired_at"])
            after = {"retired_at": self._get("keys", key)["retired_at"]}
        elif after["retired_at"] is None:
            self.store.restore_alias(key, log_event=False)
        else:
            now = None if after["retired_at"] is True else lambda: after["retired_at"]
            self.store.retire_alias(key, now=now, log_event=False)
            after = {"retired_at": self.store.get_alias(key)["retired_at"]}
        payload["changes"].append(dict(family=family, id=key, before=before, after=after))
        self._touch(payload, family, (row["subject_key"], row["owner_entity_id"]) if family == "keys" else key)
        self._touch(payload, "entities", row.get("entity_id"))

    def _entities(self):
        void = _history(self.store)[2]["entities"]
        return {r["entity_id"]: r for r in _records(self.store, "entities") if r["entity_id"] not in void}

    def _kind(self, entity):
        return _history(self.store)[3].get(entity["entity_id"], entity["kind"])

    def _new_entity(self, p, display_name, kind):
        return self._created(p, "entities", self.store.create_entity(display_name, kind, log_event=False))

    @_operation
    def create_entity(self, p, display_name, kind="person"):
        return self._new_entity(p, display_name, kind)

    def _alias(self, p, entity_id, alias_display, alias_kind="name", *, platform=None,
               verified=False, owner_entity_id=None, source_store_ids=()):
        self._get("entities", entity_id)
        if owner_entity_id is not None:
            self._get("entities", owner_entity_id)
        if alias_kind == "relation_phrase" and owner_entity_id is None:
            raise EntityOpRefused("missing_owner")
        norm = normalize_alias(alias_display)
        if alias_kind == "platform_id":
            if not platform:
                raise EntityOpRefused("invalid_platform_id")
            norm = _platform_id(normalize_platform_id(platform, alias_display))
            if not verified:
                raise EntityOpRefused("unverified_platform_id")
            holder = self.holder_for_platform_id(norm)
            if holder is not None:
                raise EntityOpRefused("platform_id_held", holder=holder)
        key = self.store.add_alias(entity_id, alias_display, alias_kind, alias_norm=norm,
            owner_entity_id=owner_entity_id, source_store_ids=source_store_ids, log_event=False)
        self._touch(p, "entities", entity_id, owner_entity_id)
        return self._created(p, "aliases", key)

    @_operation
    def add_alias(self, p, entity_id, alias_display, alias_kind="name", **options):
        return self._alias(p, entity_id, alias_display, alias_kind, **options)

    @_operation
    def retire_alias(self, p, alias_id):
        row = self._get("aliases", alias_id)
        if row["retired_at"] is not None:
            raise EntityOpRefused("not_active")
        self._project(p, "aliases", alias_id, retired_at=True)

    @_operation
    def add_note(self, p, text, *, entity_id=None, **options):
        for key in (entity_id, options.get("about_entity_id"), options.get("asserted_by_entity_id")):
            if key is not None:
                self._get("entities", key)
                self._touch(p, "entities", key)
        for field in ("superseded_by", "retracts_note_id"):
            if options.get(field) is not None:
                self._get("notes", options[field])
                self._touch(p, "notes", options[field])
        superseded = options.pop("superseded_by", None)
        key = self.store.add_note(text, entity_id=entity_id, log_event=False, **options)
        self._created(p, "notes", key)
        if superseded is not None:
            self._project(p, "notes", key, superseded_by=superseded)
        return key

    @_operation
    def change_kind(self, p, entity_id, kind):
        entity = self._get("entities", entity_id)
        if kind not in {"person", "agent", "org", "project", "place", "thing", "topic", "other"}:
            raise ValueError("invalid entity kind")
        p["kind"] = dict(entity_id=entity_id, before=self._kind(entity), after=kind)
        self._touch(p, "entities", entity_id)

    @_operation
    def merge(self, p, into, from_entity, *, allow_kind_mismatch=False):
        entities = self._entities()
        self._touch(p, "entities", into, from_entity)
        into, source = _canonical(entities, into), _canonical(entities, from_entity)
        if into == source:
            raise EntityOpRefused("same_canonical")
        if any(entities[key]["status"] == "forgotten" for key in (into, source)):
            raise EntityOpRefused("forgotten")
        if not allow_kind_mismatch and self._kind(entities[into]) != self._kind(entities[source]):
            raise EntityOpRefused("kind_mismatch")
        self._touch(p, "entities", into, source, from_entity)
        self._project(p, "entities", source, status="merged", merged_into=into)

    @_operation
    def split(self, p, entity_id, alias_ids, note_ids, display_name, kind=None):
        entity = self._get("entities", entity_id)
        if entity["status"] != "active":
            raise EntityOpRefused("not_active")
        aliases = [self._get("aliases", key) for key in dict.fromkeys(alias_ids)]
        notes = [self._get("notes", key) for key in dict.fromkeys(note_ids)]
        if not aliases and not notes:
            raise EntityOpRefused("empty_selection")
        if any(r["entity_id"] != entity_id or r.get("retired_at", r.get("superseded_by")) is not None
               for r in aliases + notes):
            raise EntityOpRefused("foreign_row")
        new = self._new_entity(p, display_name, kind or self._kind(entity))
        self._touch(p, "entities", entity_id)
        for row in aliases:
            self._project(p, "aliases", row["alias_id"], retired_at=True)
            options = {k: row[k] for k in ("alias_norm", "owner_entity_id")}
            options["source_store_ids"] = json.loads(row["source_store_ids"])
            key = self.store.add_alias(new, row["alias_display"], row["alias_kind"], log_event=False, **options)
            self._created(p, "aliases", key)
            self._touch(p, "entities", row["owner_entity_id"])
        for row in notes:
            options = {k: v for k, v in row.items() if k not in {"note_id", "scope_key", "created_at", "entity_id"}}
            for field in ("cites", "cites_file_revisions", "retracts_store_ids"):
                options[field] = json.loads(options[field])
            key = self.store.add_note(entity_id=new, log_event=False, **options)
            self._created(p, "notes", key)
            self._touch(p, "entities", row["about_entity_id"], row["asserted_by_entity_id"])
            self._touch(p, "notes", row["retracts_note_id"])
            self._project(p, "notes", row["note_id"], superseded_by=key)
        return new

    def holder_for_platform_id(self, pid):
        pid = _platform_id(pid)
        with _read_snapshot(self.store):
            aliases = self.store.aliases_by_norm(pid)
            return next((_canonical(self._entities(), a["entity_id"]) for a in aliases
                         if a["alias_kind"] == "platform_id"), None)

    def lookup_alias(self, text, access_scope=None, principal_is_owner=False):
        return lookup_alias(self.store, text, access_scope, principal_is_owner)

    def attach_or_create_for_platform_id(self, pid, display_name, *, verified=False):
        with self.store.transaction():
            pid = _platform_id(pid)
            if not verified:
                raise EntityOpRefused("unverified_platform_id")
            holder = self.holder_for_platform_id(pid)
            if holder is not None:
                return holder
            platform, identifier = pid.split(":", 1)
            p = {"created": {f: [] for f in FAMILIES}, "changes": [], "touches": {f: [] for f in FAMILIES}}
            entity = self._new_entity(p, display_name, "person")
            self._alias(p, entity, identifier, "platform_id", platform=platform, verified=True)
            self.store.append_event("create_entity", p)
            return entity

    @_operation
    def override_key(self, p, subject_key, entity_id, *, scope_store_ids=None, owner_entity_id=None):
        entity_id = _canonical(self._entities(), entity_id)
        compatible = _NAMESPACE_KINDS.get(subject_key.partition(":")[0])
        if compatible is None:
            raise EntityOpRefused("unbridged_namespace")
        if self._kind(self._get("entities", entity_id)) not in {compatible, "other", "topic"}:
            raise EntityOpRefused("kind_mismatch")
        if owner_entity_id is not None:
            self._get("entities", owner_entity_id)
        view = resolved_view(self.store)
        if owner_entity_id is None and (_relation_groups(view, subject_key.partition(":")[2]) or any(
                r["subject_key"] == subject_key and r["owner_entity_id"] is not None for r in view.get("keys", []))):
            raise EntityOpRefused("missing_owner")
        if scope_store_ids is None and len(_key_candidates(view, subject_key, owner_entity_id)) > 1:
            raise EntityOpRefused("ambiguous_key")
        sources = {source for key, source in _key_sources(self.store) if key == subject_key}
        if scope_store_ids is None and owner_entity_id is not None:
            owner_rows = _owner_alias_rows(self.store, subject_key, owner_entity_id)[:1]
            sources &= {source for row in _active_keys(self.store) + owner_rows
                        if (row["subject_key"], row["owner_entity_id"]) == (subject_key, owner_entity_id)
                        for source in json.loads(row["source_store_ids"])}
        cited = sorted(sources if scope_store_ids is None else sources.intersection(scope_store_ids))
        p["override"] = dict(subject_key=subject_key, entity_id=entity_id, owner_entity_id=owner_entity_id,
                             scope_store_ids=cited, global_override=scope_store_ids is None)
        self._touch(p, "entities", entity_id, owner_entity_id)
        self._touch(p, "keys", (subject_key, owner_entity_id))
        if scope_store_ids is None:
            for row in _active_keys(self.store):
                if (row["subject_key"], row["owner_entity_id"]) == (subject_key, owner_entity_id):
                    self._project(p, "keys", row["key_id"], retired_at=True)
            p["created"]["keys"].append(self.store.add_key(subject_key, entity_id, "override", cited,
                                                        owner_entity_id=owner_entity_id))

    def undo(self, event_id):
        with self.store.transaction():
            events, undone, _, _ = _history(self.store)
            target = next((e for e in events if e["event_id"] == event_id), None)
            if target is None:
                raise EntityOpRefused("not_found")
            if target["op"] == "undo":
                raise EntityOpRefused("is_undo")
            if event_id in undone:
                raise EntityOpRefused("already_undone")
            inverse = INVERSE_REGISTRY.get(target["op"])
            if inverse is None or "touches" not in target["payload"]:
                raise EntityOpRefused("unsupported_op")
            p = target["payload"]
            blockers = [e["event_id"] for e in reversed(events) if e["event_id"] > event_id
                and e["op"] != "undo" and e["event_id"] not in undone
                and ("touches" not in e["payload"] or any(
                    {tuple(k) if isinstance(k, list) else k for k in p["touches"].get(f, [])} &
                    {tuple(k) if isinstance(k, list) else k for k in e["payload"]["touches"].get(f, [])}
                    for f in FAMILIES))]
            if blockers:
                raise EntityOpRefused("dependent_events", blocking_event_ids=blockers)
            override = p.get("override")
            if override and override["global_override"]:
                # Rebuild may replace this event's row after canonical/source churn.
                owned = [r["key_id"] for r in _active_keys(self.store)
                         if (r["subject_key"], r["owner_entity_id"], r["method"]) ==
                         (override["subject_key"], override["owner_entity_id"], "override")
                         and r["entity_id"] == _canonical(self._entities(), override["entity_id"])
                         and set(json.loads(r["source_store_ids"])) <= set(override["scope_store_ids"])]
                p = dict(p, created=dict(p["created"], keys=list(set(p["created"].get("keys", []) + owned))))
            for change in p["changes"]:
                if change["family"] == "keys" and change["before"]["retired_at"] is None:
                    row = self._get("keys", change["id"])
                    if any(k["subject_key"] == row["subject_key"] and k["owner_entity_id"] == row["owner_entity_id"]
                           and k["key_id"] not in p["created"].get("keys", []) for k in _active_keys(self.store)):
                        raise EntityOpRefused("key_held")
            for change in p["changes"]:
                if change["family"] == "aliases" and change["before"]["retired_at"] is None:
                    alias = self._get("aliases", change["id"])
                    holders = self.store.aliases_by_norm(alias["alias_norm"])
                    if alias["alias_kind"] == "platform_id":
                        for holder in holders:
                            if holder["alias_kind"] == "platform_id" and holder["alias_id"] not in p["created"]["aliases"]:
                                raise EntityOpRefused("platform_id_held", holder=holder["entity_id"])
            self._undo_payload = {"created": {f: [] for f in FAMILIES}, "changes": [],
                                  "touches": {f: [] for f in FAMILIES}}
            try:
                inverse(self, p)
            except sqlite3.IntegrityError as error:
                if target["op"] != "override_key":
                    raise
                raise EntityOpRefused("key_held") from error
            return self.store.append_event("undo", self._undo_payload, target_event_id=event_id)


def replay_projection(store):
    """Recompute the four stored fields using only event payloads, including undo."""
    result = {f: {} for f in FAMILIES if f != "keys"}
    for event in _history(store)[0]:
        p = event["payload"]
        if not all(field in p for field in ("created", "changes", "touches")):
            raise EntityIntegrityError("event lacks operations payload")
        for family, keys in p.get("created", {}).items():
            if family == "keys":
                continue
            for key in keys:
                result[family][key] = ("active", None) if family == "entities" else (None,)
        for change in p.get("changes", []):
            if change["family"] == "keys":
                continue
            fields = ("status", "merged_into") if change["family"] == "entities" else (
                "retired_at" if change["family"] == "aliases" else "superseded_by",)
            result[change["family"]][change["id"]] = tuple(change["after"][f] for f in fields)
    return result


@contextmanager
def _read_snapshot(store):
    # SAVEPOINT also works on read-only stores and inside an operation transaction.
    with store._lock:
        store.connection.execute("SAVEPOINT entity_resolved_read")
        try:
            yield
        finally:
            store.connection.execute("RELEASE entity_resolved_read")


def _visible_rows(store, family, void, access):
    table, key = FAMILIES[family]
    rows = store._read(table, **access)
    active = "retired_at" if family == "aliases" else "superseded_by"
    return [r for r in rows if r[key] not in void[family] and r[active] is None]


def _cards(entities, canonical, kinds, aliases, notes):
    cards = {}
    for key in sorted(set(canonical.values())):
        row = entities[key]
        if row["status"] == "active":
            cards[key] = dict(display_name=row["display_name"], kind=kinds.get(key, row["kind"]),
                              aliases=[], notes=[])
    # Cards expose record content, never event payloads or hidden reference ids.
    for rows, family, fields in [
            (aliases, "aliases", ("alias_id", "alias_display", "alias_norm", "alias_kind")),
            (notes, "notes", ("note_id", "text", "kind"))]:
        for row in rows:
            key = canonical.get(row["entity_id"])
            if key in cards:
                cards[key][family].append({field: row[field] for field in fields})
    return cards


def resolved_view(store, access_scope=None, principal_is_owner=False, *, _platform_norm=None):
    """Reduce trusted scoped history, then emit only S-visible source records.

    Entity eligibility is supported by visible aliases/notes on it or on entities
    merged into it. Hidden intermediate nodes never appear in the returned map.
    No derived state is retained between calls or shared across store instances.
    """
    access = dict(access_scope=access_scope, principal_is_owner=principal_is_owner)
    with _read_snapshot(store):
        _, _, void, kinds = _history(store)
        entities = {r["entity_id"]: r for r in _records(store, "entities")
                    if r["entity_id"] not in void["entities"]}
        canonical = {key: _canonical(entities, key) for key in entities}
        aliases = _visible_rows(store, "aliases", void, access)
        if _platform_norm is not None:
            aliases = [a for a in aliases if a["alias_kind"] != "platform_id" or a["alias_norm"] == _platform_norm]
        notes = _visible_rows(store, "notes", void, access)
        if access_scope is not None:
            visible = {r["entity_id"] for r in aliases + notes if r["entity_id"] in entities}
            # A visible merged descendant also supports every ancestor's identity.
            for key in tuple(visible):
                while entities[key]["merged_into"] is not None:
                    key = entities[key]["merged_into"]
                    visible.add(key)
            canonical = {key: target for key, target in canonical.items() if key in visible}
        candidates, relations = {}, {}
        for row in aliases:
            target = canonical.get(row["entity_id"])
            if target is not None and entities[target]["status"] == "active":
                group = candidates
                if row["alias_kind"] == "relation_phrase":
                    owner = row["owner_entity_id"]
                    if access_scope is not None and owner not in canonical:
                        continue
                    group = relations.setdefault(row["alias_norm"], {})
                    group.setdefault(owner, set()).add(target)
                else:
                    group.setdefault(row["alias_norm"], set()).add(target)
        view = dict(canonical=canonical,
                    aliases={text: dict(candidates=sorted(ids), ambiguous=len(ids) > 1)
                             for text, ids in sorted(candidates.items())},
                    cards=_cards(entities, canonical, kinds, aliases, notes))
        if relations:
            view["relations"] = {text: {owner: dict(candidates=sorted(ids), ambiguous=len(ids) > 1)
                                      for owner, ids in groups.items()} for text, groups in relations.items()}
        keys = [dict(subject_key=r["subject_key"], owner_entity_id=r["owner_entity_id"],
                     entity_id=canonical[r["entity_id"]], method=r["method"],
                     source_store_ids=json.loads(r["source_store_ids"]))
                for r in store._read("lcm_entity_keys", "t.retired_at IS NULL", **access)
                if r["entity_id"] in view["cards"] or r["entity_id"] in canonical
                if access_scope is None or r["owner_entity_id"] is None or r["owner_entity_id"] in canonical]
        if access_scope is not None:
            visible_keys = []
            for row in keys:
                if row["method"] == "alias_match" and row["subject_key"].partition(":")[0] not in {"user", "assistant"}:
                    candidates = _key_candidates(view, row["subject_key"], row["owner_entity_id"])
                    if len(candidates) != 1:
                        continue
                    row["entity_id"] = candidates[0]
                visible_keys.append(row)
            keys = visible_keys
        if keys:
            view["keys"] = sorted(keys, key=lambda r: (r["subject_key"], r["owner_entity_id"] or ""))
        return view


def lookup_alias(store, text, access_scope=None, principal_is_owner=False):
    # Names use S normalization; platform identifiers retain their exact bytes.
    if ":" in text:
        text = normalize_platform_id(*text.split(":", 1))
    view = resolved_view(store, access_scope, principal_is_owner, _platform_norm=text)
    return view["aliases"].get(text, view["aliases"].get(
        normalize_alias(text), dict(candidates=[], ambiguous=False)))


_NAMESPACE_KINDS = dict(person="person", user="person", assistant="agent", organization="org",
                        project="project", place="place", object="thing", account="thing", document="thing")


def _key_sources(store, access_scope=None, principal_is_owner=False):
    from . import assertion_store
    from .db_bootstrap import verify_assertion_schema
    if access_scope is not None and not isinstance(principal_is_owner, bool):
        raise TypeError("principal_is_owner must be a bool in principal mode")
    present = store.connection.execute(
        "SELECT 1 FROM sqlite_master WHERE name='lcm_assertions' AND type='table'").fetchone()
    if not present:
        return []
    if verify_assertion_schema(store.connection):
        raise EntityIntegrityError("assertion family is not verified")
    return [tuple(r) for r in store.connection.execute(f"""
        SELECT DISTINCT a.subject_key,a.source_store_id FROM lcm_assertions a
        JOIN lcm_assertion_sources s ON s.source_store_id=a.source_store_id
          AND s.extraction_version=a.extraction_version AND s.source_content_sha256=a.source_content_sha256
        LEFT JOIN messages m ON m.store_id=a.source_store_id
        WHERE a.extraction_version=:version AND s.invalidated_at IS NULL AND m.store_id IS NOT NULL
          AND (:principal IS NULL OR :principal!='') AND ({store._visible('json_array(a.source_store_id)')})
        ORDER BY a.subject_key,a.source_store_id""", dict(version=assertion_store.CURRENT_EXTRACTION_VERSION,
            scope=store.scope_key, principal=access_scope, owner=principal_is_owner))]


def _relation_groups(view, value):
    relations = view.get("relations", {})
    norm = normalize_alias(value)
    return relations.get(norm, relations.get("my " + norm, {}))


def _key_candidates(view, subject_key, owner=None):
    namespace, _, value = subject_key.partition(":")
    compatible = _NAMESPACE_KINDS.get(namespace)
    if namespace in {"user", "assistant"}:
        return []
    relations = _relation_groups(view, value)
    if relations or owner is not None:
        candidates = relations.get(owner, {}).get("candidates", [])
    else:
        exact = normalize_platform_id(*value.split(":", 1)) if ":" in value else None
        candidates = [key for key, card in view["cards"].items() if any(
            a["alias_norm"] == (exact if a["alias_kind"] == "platform_id" else normalize_alias(value))
            for a in card["aliases"] if a["alias_kind"] != "relation_phrase")]
    return [key for key in candidates if view["cards"][key]["kind"] == compatible]


def _active_keys(store):
    # Index maintenance must also see rows whose cited messages were deleted.
    return [r for r in _records(store, "keys") if r["retired_at"] is None]


def _owner_alias_rows(store, subject, owner):
    return [dict(r) for r in store.connection.execute(
        "SELECT * FROM lcm_entity_keys WHERE scope_key=? AND subject_key=? "
        "AND owner_entity_id=? AND method='alias_match' ORDER BY rowid DESC",
        (store.scope_key, subject, owner))]


def _overrides(events, undone):
    return [e["payload"]["override"] for e in events
            if e["op"] == "override_key" and e["event_id"] not in undone]


def rebuild_entity_keys(store, *, sender_lookup=None, profile_entities=None):
    """Reconcile the derived index without appending memory events."""
    with store.transaction():
        sources = _key_sources(store)
        view, desired = resolved_view(store), {}
        profiles = profile_entities(store) if profile_entities else None
        for subject, source in sources:
            namespace, _, value = subject.partition(":")
            owner, candidates = None, _key_candidates(view, subject)
            if namespace in {"user", "assistant"}:
                candidates = [profiles[namespace == "assistant"]] if profiles else []
            if namespace == "user" or _relation_groups(view, value):
                sender = sender_lookup(source) if sender_lookup else None
                if sender and sender[1] and sender[2] and sender[0] in view["canonical"]:
                    owner = sender[0]
                    candidates = [view["canonical"][owner]] if namespace == "user" else _key_candidates(view, subject, owner)
                elif namespace != "user" or not profiles:
                    candidates = []
            candidates = [key for key in candidates if key in view["cards"] and
                          view["cards"][key]["kind"] == _NAMESPACE_KINDS.get(namespace)]
            if len(candidates) == 1:
                entry = desired.setdefault((subject, owner), [candidates[0], "alias_match", set()])
                entry[2].add(source)
        events, undone, _, _ = _history(store)
        alias_desired = desired.copy()
        writes = 0
        for override in _overrides(events, undone):
            if not override["global_override"]:
                continue
            subject, owner = override["subject_key"], override["owner_entity_id"]
            bound = {source for key, source in sources if key == subject} & set(override["scope_store_ids"])
            target = view["canonical"].get(override["entity_id"])
            if bound and target in view["cards"]:
                alias = alias_desired.get((subject, owner))
                # Keep the owner/source derivation even while the unique active
                # slot holds an override. Uncited evidence still uses aliases.
                if owner is not None and alias and not any(
                        r["entity_id"] == alias[0] and set(json.loads(r["source_store_ids"])) == alias[2]
                        for r in _owner_alias_rows(store, subject, owner)[:1]):
                    held = [r for r in _active_keys(store)
                            if (r["subject_key"], r["owner_entity_id"]) == (subject, owner)]
                    for row in held:
                        store.retire_key(row["key_id"])
                    key = store.add_key(subject, alias[0], "alias_match", sorted(alias[2]), owner_entity_id=owner)
                    store.retire_key(key)
                    for row in held:
                        store.restore_key(row["key_id"])
                    writes += 2 + 2 * len(held)
                desired[(subject, owner)] = [target, "override", bound]
        for row in _active_keys(store):
            identity = (row["subject_key"], row["owner_entity_id"])
            wanted = desired.get(identity)
            if wanted and (row["entity_id"], row["method"], set(json.loads(row["source_store_ids"]))) == tuple(wanted):
                desired.pop(identity)
            else:
                store.retire_key(row["key_id"])
                writes += 1
        for (subject, owner), (target, method, cited) in desired.items():
            try:
                store.add_key(subject, target, method, sorted(cited), owner_entity_id=owner)
            except sqlite3.IntegrityError as error:
                raise EntityOpRefused("key_held") from error
            writes += 1
        return writes


def resolve_key(store, subject_key, store_id, *, owner_entity_id=None,
                access_scope=None, principal_is_owner=False):
    """Resolve one current source; overrides bind reviewed evidence only."""
    with _read_snapshot(store):
        if owner_entity_id is not None:
            EntityOps(store)._get("entities", owner_entity_id)
        if (subject_key, store_id) not in _key_sources(store, access_scope, principal_is_owner):
            return None
        view = resolved_view(store, access_scope, principal_is_owner)
        events, undone, _, _ = _history(store)
        for override in reversed(_overrides(events, undone)):
            if (override["subject_key"], override["owner_entity_id"]) == (subject_key, owner_entity_id) and store_id in override["scope_store_ids"]:
                target = view["canonical"].get(override["entity_id"])
                return target if target in view["cards"] else None
        rows = [r for r in view.get("keys", []) if (r["subject_key"], r["owner_entity_id"], r["method"]) ==
                (subject_key, owner_entity_id, "alias_match") and store_id in r["source_store_ids"]]
        if owner_entity_id is not None and not rows:
            derived = _owner_alias_rows(store, subject_key, owner_entity_id)
            rows = [r for r in derived[:1] if store_id in json.loads(r["source_store_ids"])
                    and r["entity_id"] in view["cards"]]
        relation = any(r["alias_kind"] == "relation_phrase" and r["retired_at"] is None
                       and r["alias_norm"] in {normalize_alias(subject_key.partition(":")[2]),
                                               "my " + normalize_alias(subject_key.partition(":")[2])}
                       for r in _records(store, "aliases"))
        if relation and owner_entity_id is None:
            return None
        candidates = _key_candidates(view, subject_key, owner_entity_id)
        if owner_entity_id is not None and not rows:
            return None
        if subject_key.partition(":")[0] in {"user", "assistant"}:
            target = view["canonical"].get(rows[0]["entity_id"]) if rows else None
            return target if target in view["cards"] and view["cards"][target]["kind"] == _NAMESPACE_KINDS[
                subject_key.partition(":")[0]] else None
        return candidates[0] if len(candidates) == 1 else None
