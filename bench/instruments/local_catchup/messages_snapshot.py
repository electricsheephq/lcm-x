"""Physical messages projection for pre-S0 PIN contracts."""
_EXCLUDED = frozenset({"sender_id", "sender_source"})


def messages_snapshot_columns(conn):
    return ", ".join(f'"{row[1]}"' for row in conn.execute("PRAGMA table_info(messages)")
                     if row[1] not in _EXCLUDED)
