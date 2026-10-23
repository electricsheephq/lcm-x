"""Default-off entity operations. Source writes belong to EntityStore."""

from functools import wraps
from contextlib import contextmanager
import json

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
            "notes": ("lcm_memory_notes", "note_id")}


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
            result = method(self, p, *args, **kwargs)
            event = self.store.append_event(method.__name__, p)
            return event if result is None else result
    return run


def _inverse(ops, payload):
    # Retire copies before restoring originals, including platform ids.
    for alias in payload["created"]["aliases"]:
        ops._project(payload=ops._undo_payload, family="aliases", key=alias, retired_at=True)
    changes = list(reversed(payload["changes"]))
    for change in sorted(changes, key=lambda c: c["family"] != "aliases"):
        ops._project(ops._undo_payload, change["family"], change["id"], **change["before"])


# W2 may register an inverse(ops, payload) using the same payload/touch contract.
INVERSE_REGISTRY = dict.fromkeys(
    ("create_entity", "add_alias", "retire_alias", "change_kind", "merge", "split", "add_note"), _inverse)


class EntityOps:
    """One profile's transactional operations; ids returned by creators are opaque."""

    def __init__(self, store):
        self.store = store

    def _get(self, family, key):
        getter = {"entities": self.store.get_entity, "aliases": self.store.get_alias,
                  "notes": self.store.get_note}[family]
        row = getter(key)
        if row is None or key in _history(self.store)[2][family]:
            raise EntityOpRefused("not_found")
        return row

    def _touch(self, p, family, *keys):
        for key in keys:
            if key is not None and key not in p["touches"][family]:
                p["touches"][family].append(key)

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
        elif after["retired_at"] is None:
            self.store.restore_alias(key, log_event=False)
        else:
            now = None if after["retired_at"] is True else lambda: after["retired_at"]
            self.store.retire_alias(key, now=now, log_event=False)
            after = {"retired_at": self.store.get_alias(key)["retired_at"]}
        payload["changes"].append(dict(family=family, id=key, before=before, after=after))
        self._touch(payload, family, key)
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
            if not verified:
                raise EntityOpRefused("unverified_platform_id")
            norm = normalize_platform_id(platform, alias_display)
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
        with self.store.transaction():
            aliases = self.store.aliases_by_norm(pid)
            return next((_canonical(self._entities(), a["entity_id"]) for a in aliases
                         if a["alias_kind"] == "platform_id"), None)

    def lookup_alias(self, text, access_scope=None, principal_is_owner=False):
        return lookup_alias(self.store, text, access_scope, principal_is_owner)

    def attach_or_create_for_platform_id(self, pid, display_name, *, verified=False):
        with self.store.transaction():
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
                and any(set(p["touches"][f]) & set(e["payload"].get("touches", {}).get(f, [])) for f in FAMILIES)]
            if blockers:
                raise EntityOpRefused("dependent_events", blocking_event_ids=blockers)
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
            inverse(self, p)
            return self.store.append_event("undo", self._undo_payload, target_event_id=event_id)


def replay_projection(store):
    """Recompute the four stored fields using only event payloads, including undo."""
    result = {f: {} for f in FAMILIES}
    for event in _history(store)[0]:
        p = event["payload"]
        for family, keys in p.get("created", {}).items():
            for key in keys:
                result[family][key] = ("active", None) if family == "entities" else (None,)
        for change in p.get("changes", []):
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


def resolved_view(store, access_scope=None, principal_is_owner=False):
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
        notes = _visible_rows(store, "notes", void, access)
        if access_scope is not None:
            visible = {r["entity_id"] for r in aliases + notes if r["entity_id"] in entities}
            # A visible merged descendant also supports every ancestor's identity.
            for key in tuple(visible):
                while entities[key]["merged_into"] is not None:
                    key = entities[key]["merged_into"]
                    visible.add(key)
            canonical = {key: target for key, target in canonical.items() if key in visible}
        candidates = {}
        for row in aliases:
            target = canonical.get(row["entity_id"])
            if target is not None and entities[target]["status"] == "active":
                candidates.setdefault(row["alias_norm"], set()).add(target)
        return dict(canonical=canonical,
                    aliases={text: dict(candidates=sorted(ids), ambiguous=len(ids) > 1)
                             for text, ids in sorted(candidates.items())},
                    cards=_cards(entities, canonical, kinds, aliases, notes))


def lookup_alias(store, text, access_scope=None, principal_is_owner=False):
    view = resolved_view(store, access_scope, principal_is_owner)
    # Names use S normalization; platform identifiers retain their exact bytes.
    return view["aliases"].get(text, view["aliases"].get(
        normalize_alias(text), dict(candidates=[], ambiguous=False)))
