-- Synthetic #2558 fixture: real v0.29.2 engine 261d41ea, public session-start/ingest/shutdown; sqlite3 .dump.
-- Clock and source-file stat timestamps frozen at 1762700100; only absolute home/temp paths substituted; FTS rebuilt after substitution.
/* WARNING: Script requires that SQLITE_DBCONFIG_DEFENSIVE be disabled */
PRAGMA foreign_keys=OFF;
BEGIN TRANSACTION;
CREATE TABLE messages (
                store_id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                source TEXT DEFAULT '',
                conversation_id TEXT DEFAULT '',
                role TEXT NOT NULL,
                content TEXT,
                tool_call_id TEXT,
                tool_calls TEXT,
                tool_name TEXT,
                timestamp REAL NOT NULL,
                token_estimate INTEGER DEFAULT 0,
                pinned INTEGER DEFAULT 0,
                ingested_at REAL,
                observed_at REAL,
                observed_at_source TEXT,
                sender_id TEXT,
                sender_source TEXT
            );
INSERT INTO messages VALUES(1,'u1-session','cli','u1-conv','user','Please dump the big report.',NULL,NULL,NULL,1762700100.0,11,0,1762700100.0,1762700001.0,'host_message_timestamp',NULL,NULL);
INSERT INTO messages VALUES(2,'u1-session','cli','u1-conv','assistant','',NULL,'[{"id": "call_spill", "type": "function", "function": {"name": "dump", "arguments": "{}"}}]',NULL,1762700100.0,10,0,1762700100.0,1762700002.0,'host_message_timestamp',NULL,NULL);
INSERT INTO messages VALUES(3,'u1-session','cli','u1-conv','tool','[Externalized tool output: tool=dump; tool_call_id=call_spill; chars=5785; bytes=7585; read it with lcm_expand(externalized_ref="20251109_145500_call_spill_a01d707efe08_18765e66da0ba800.json"); ref=20251109_145500_call_spill_a01d707efe08_18765e66da0ba800.json]','call_spill',NULL,NULL,1762700100.0,70,0,1762700100.0,1762700003.0,'host_message_timestamp',NULL,NULL);
INSERT INTO messages VALUES(4,'u1-session','cli','u1-conv','assistant','The report is saved; the needle is at the top.',NULL,NULL,NULL,1762700100.0,16,0,1762700100.0,1762700004.0,'host_message_timestamp',NULL,NULL);
CREATE TABLE metadata (
                key TEXT PRIMARY KEY,
                value TEXT
            );
INSERT INTO metadata VALUES('fts_integrity_checked_at:messages_fts','1762700100.0');
INSERT INTO metadata VALUES('schema_version','5');
INSERT INTO metadata VALUES('fts_stem_state','ready');
INSERT INTO metadata VALUES('fts_integrity_checked_at:nodes_fts','1762700100.0');
INSERT INTO metadata VALUES('ignored_active_replay_placeholder_hash_counts:u1-session','{}');
INSERT INTO metadata VALUES('ignored_active_replay_placeholder_hash_ordinals:u1-session','{}');
PRAGMA writable_schema=ON;
INSERT INTO sqlite_schema(type,name,tbl_name,rootpage,sql)VALUES('table','messages_fts','messages_fts',0,'CREATE VIRTUAL TABLE "messages_fts" USING fts5(
                        "content",
                        content="messages",
                        content_rowid="store_id"
                    )');
CREATE TABLE IF NOT EXISTS 'messages_fts_data'(id INTEGER PRIMARY KEY, block BLOB);
INSERT INTO messages_fts_data VALUES(1,x'0433');
INSERT INTO messages_fts_data VALUES(10,x'000000000101010001010101');
INSERT INTO messages_fts_data VALUES(137438953473,x'0000014807303134353530300304180a020f38373635653636646130626138303003041c0a010832303235313130390304170a01043537383503020d01043735383503020f010c61303164373037656665303803041b0a020174040209010362696701020502047974657303020e010463616c6c03080804110a02046861727303020c010464756d700102030202060106657870616e64030214030a7465726e616c697a656403040215010269640302090201730404040602017403021101046a736f6e03041d0a01036c636d03021301066e6565646c6504020701066f75747075740302040106706c656173650102020104726561640302100301660304160a0304706f727401020603020301057361766564040205020470696c6c03060b110a0103746865010204030602060602036f6f6c030603040403017004020b010477697468030212040c150e0909120608090c090c0b100707060a080b0b0b09070c0a0b0d0a06');
CREATE TABLE IF NOT EXISTS 'messages_fts_idx'(segid, term, pgno, PRIMARY KEY(segid, term)) WITHOUT ROWID;
INSERT INTO messages_fts_idx VALUES(1,x'',2);
CREATE TABLE IF NOT EXISTS 'messages_fts_docsize'(id INTEGER PRIMARY KEY, sz BLOB);
INSERT INTO messages_fts_docsize VALUES(1,x'05');
INSERT INTO messages_fts_docsize VALUES(2,x'00');
INSERT INTO messages_fts_docsize VALUES(3,x'24');
INSERT INTO messages_fts_docsize VALUES(4,x'0a');
CREATE TABLE IF NOT EXISTS 'messages_fts_config'(k PRIMARY KEY, v) WITHOUT ROWID;
INSERT INTO messages_fts_config VALUES('version',4);
CREATE TABLE lcm_migration_state (
            step_name TEXT PRIMARY KEY,
            completed_at REAL NOT NULL
        );
INSERT INTO lcm_migration_state VALUES('v2_external_content_fts_triggers',1794288821.0);
INSERT INTO lcm_migration_state VALUES('v3_lifecycle_state',1794288821.0);
INSERT INTO lcm_migration_state VALUES('v4_lifecycle_debt_columns',1794288821.0);
INSERT INTO lcm_migration_state VALUES('v5_message_conversation_id',1794288821.0);
INSERT INTO lcm_migration_state VALUES('messages_ingested_at_backfill_v1',1794288821.0);
INSERT INTO lcm_migration_state VALUES('fts_stem_v1',1794288821.0);
INSERT INTO lcm_migration_state VALUES('temporal_rollups_v1',1794288821.0);
INSERT INTO lcm_migration_state VALUES('embeddings_v1',1794288821.0);
INSERT INTO lcm_migration_state VALUES('chunk_vectors_v1',1794288821.0);
CREATE TABLE lcm_lifecycle_state (
            conversation_id TEXT PRIMARY KEY,
            current_session_id TEXT,
            last_finalized_session_id TEXT,
            current_frontier_store_id INTEGER NOT NULL DEFAULT 0,
            last_finalized_frontier_store_id INTEGER NOT NULL DEFAULT 0,
            debt_kind TEXT,
            debt_size_estimate INTEGER NOT NULL DEFAULT 0,
            current_bound_at REAL,
            last_finalized_at REAL,
            debt_updated_at REAL,
            last_maintenance_attempt_at REAL,
            last_rollover_at REAL,
            last_reset_at REAL,
            updated_at REAL NOT NULL DEFAULT (strftime('%s','now'))
        );
INSERT INTO lcm_lifecycle_state VALUES('u1-conv','u1-session',NULL,0,0,NULL,0,1762700100.0,NULL,NULL,NULL,NULL,NULL,1762700100.0);
INSERT INTO sqlite_schema(type,name,tbl_name,rootpage,sql)VALUES('table','messages_fts_stem','messages_fts_stem',0,'CREATE VIRTUAL TABLE messages_fts_stem USING fts5(content, content=''messages'', content_rowid=''store_id'', tokenize=''porter unicode61'')');
CREATE TABLE IF NOT EXISTS 'messages_fts_stem_data'(id INTEGER PRIMARY KEY, block BLOB);
INSERT INTO messages_fts_stem_data VALUES(1,x'0433');
INSERT INTO messages_fts_stem_data VALUES(10,x'000000000101010001010101');
INSERT INTO messages_fts_stem_data VALUES(137438953473,x'0000013d07303134353530300304180a020f38373635653636646130626138303003041c0a010832303235313130390304170a01043537383503020d01043735383503020f010c61303164373037656665303803041b0a0201740402090103626967010205020379746503020e010463616c6c03080804110a020368617203020c010464756d700102030202060106657870616e6403021403047465726e03040215010269640302090201730404040602017403021101046a736f6e03041d0a01036c636d03021301056e6565646c04020701066f75747075740302040105706c6561730102020104726561640302100301660304160a0304706f7274010206030203010473617665040205020470696c6c03060b110a0103746865010204030602060602036f6f6c030603040403017004020b010477697468030212040c150e0909120608080c080c0b0a0707060a080a0b0a09070c090b0d0a06');
CREATE TABLE IF NOT EXISTS 'messages_fts_stem_idx'(segid, term, pgno, PRIMARY KEY(segid, term)) WITHOUT ROWID;
INSERT INTO messages_fts_stem_idx VALUES(1,x'',2);
CREATE TABLE IF NOT EXISTS 'messages_fts_stem_docsize'(id INTEGER PRIMARY KEY, sz BLOB);
INSERT INTO messages_fts_stem_docsize VALUES(1,x'05');
INSERT INTO messages_fts_stem_docsize VALUES(2,x'00');
INSERT INTO messages_fts_stem_docsize VALUES(3,x'24');
INSERT INTO messages_fts_stem_docsize VALUES(4,x'0a');
CREATE TABLE IF NOT EXISTS 'messages_fts_stem_config'(k PRIMARY KEY, v) WITHOUT ROWID;
INSERT INTO messages_fts_stem_config VALUES('version',4);
CREATE TABLE summary_nodes (
                node_id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                depth INTEGER NOT NULL DEFAULT 0,
                summary TEXT NOT NULL,
                token_count INTEGER DEFAULT 0,
                source_token_count INTEGER DEFAULT 0,
                source_ids TEXT NOT NULL DEFAULT '[]',
                source_type TEXT NOT NULL DEFAULT 'messages',
                created_at REAL NOT NULL,
                earliest_at REAL,
                latest_at REAL,
                expand_hint TEXT DEFAULT ''
            );
CREATE TABLE summary_node_provenance (
                node_id INTEGER PRIMARY KEY,
                escalation_level INTEGER NOT NULL,
                model TEXT NOT NULL DEFAULT '',
                created_at REAL NOT NULL
            );
INSERT INTO sqlite_schema(type,name,tbl_name,rootpage,sql)VALUES('table','nodes_fts','nodes_fts',0,'CREATE VIRTUAL TABLE "nodes_fts" USING fts5(
                        "summary",
                        content="summary_nodes",
                        content_rowid="node_id"
                    )');
CREATE TABLE IF NOT EXISTS 'nodes_fts_data'(id INTEGER PRIMARY KEY, block BLOB);
INSERT INTO nodes_fts_data VALUES(1,x'0000');
INSERT INTO nodes_fts_data VALUES(10,x'00000000000000');
CREATE TABLE IF NOT EXISTS 'nodes_fts_idx'(segid, term, pgno, PRIMARY KEY(segid, term)) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS 'nodes_fts_docsize'(id INTEGER PRIMARY KEY, sz BLOB);
CREATE TABLE IF NOT EXISTS 'nodes_fts_config'(k PRIMARY KEY, v) WITHOUT ROWID;
INSERT INTO nodes_fts_config VALUES('version',4);
CREATE TABLE lcm_rollups (
            rollup_id INTEGER PRIMARY KEY AUTOINCREMENT,
            period_kind TEXT NOT NULL CHECK (period_kind IN ('day', 'week', 'month')),
            period_start TEXT NOT NULL,
            scope TEXT NOT NULL,
            summary TEXT,
            token_count INTEGER,
            status TEXT NOT NULL DEFAULT 'building'
                CHECK (status IN ('building', 'ready', 'stale', 'failed')),
            built_at TEXT,
            source_fingerprint TEXT,
            error TEXT,
            generation INTEGER NOT NULL DEFAULT 0,
            lease_expires_at TEXT,
            lease_nonce TEXT NOT NULL DEFAULT '',
            failed_at TEXT,
            UNIQUE(period_kind, period_start, scope)
        );
CREATE TABLE lcm_rollup_sources (
            rollup_id INTEGER NOT NULL,
            node_id INTEGER NOT NULL,
            PRIMARY KEY(rollup_id, node_id)
        );
CREATE TABLE lcm_rollup_invalidations (
            event_id INTEGER PRIMARY KEY AUTOINCREMENT,
            node_id INTEGER,
            scope TEXT NOT NULL,
            covered_start REAL NOT NULL,
            covered_end REAL NOT NULL,
            next_day TEXT,
            operation TEXT NOT NULL CHECK(operation IN ('insert', 'delete', 'update')),
            created_at REAL NOT NULL DEFAULT (strftime('%s','now'))
        );
CREATE TABLE lcm_rollup_state (
            period_kind TEXT NOT NULL,
            scope TEXT NOT NULL DEFAULT '',
            last_build_cursor TEXT,
            last_built_at TEXT,
            PRIMARY KEY(period_kind, scope)
        );
CREATE TABLE lcm_embedding_profile (
            identity_hash TEXT PRIMARY KEY,
            provider TEXT NOT NULL,
            model_name TEXT NOT NULL,
            revision TEXT NOT NULL DEFAULT '',
            dim INTEGER CHECK(dim BETWEEN 1 AND 4096),
            dtype TEXT NOT NULL DEFAULT 'float32',
            byteorder TEXT NOT NULL DEFAULT 'little',
            task TEXT NOT NULL DEFAULT 'summary',
            registered_at TEXT,
            active INTEGER DEFAULT 1,
            archived_at TEXT NULL,
            data_version INTEGER NOT NULL DEFAULT 0
        );
CREATE TABLE lcm_embedding_meta (
            embedded_id TEXT,
            embedded_kind TEXT CHECK(embedded_kind IN ('summary')),
            identity_hash TEXT,
            embedded_at TEXT,
            source_token_count INTEGER,
            archived INTEGER DEFAULT 0,
            PRIMARY KEY(embedded_id, embedded_kind, identity_hash)
        );
CREATE TABLE lcm_embedding_vectors (
            embedded_id TEXT,
            identity_hash TEXT,
            vec BLOB NOT NULL,
            PRIMARY KEY(embedded_id, identity_hash)
        );
CREATE TABLE lcm_embedding_binary (
            embedded_id TEXT,
            identity_hash TEXT,
            bits BLOB NOT NULL,
            PRIMARY KEY(embedded_id, identity_hash)
        );
CREATE TABLE lcm_chunk_meta (
            chunk_id TEXT,
            identity_hash TEXT,
            store_id INTEGER,
            chunk_index INTEGER,
            char_start INTEGER,
            char_end INTEGER,
            token_estimate INTEGER,
            embedded_at TEXT,
            archived INTEGER DEFAULT 0,
            PRIMARY KEY(chunk_id, identity_hash)
        );
CREATE TABLE lcm_chunk_vectors (
            chunk_id TEXT,
            identity_hash TEXT,
            vec BLOB NOT NULL,
            PRIMARY KEY(chunk_id, identity_hash)
        );
CREATE TABLE lcm_chunk_binary (
            chunk_id TEXT,
            identity_hash TEXT,
            bits BLOB NOT NULL,
            PRIMARY KEY(chunk_id, identity_hash)
        );
CREATE TABLE lcm_embedding_backfill_inflight (
                embedded_id TEXT,
                identity_hash TEXT,
                lease_id TEXT,
                generation INTEGER,
                claimed_at REAL,
                state TEXT NOT NULL DEFAULT 'claimed'
                    CHECK(state IN ('claimed', 'dispatched', 'uncertain')),
                request_id TEXT,
                updated_at REAL,
                last_error TEXT,
                PRIMARY KEY(embedded_id, identity_hash)
            );
CREATE TABLE message_relations (
                    relation_id INTEGER PRIMARY KEY,
                    store_id INTEGER NOT NULL,
                    kind TEXT NOT NULL,
                    related_store_id INTEGER,
                    ordinal INTEGER,
                    observed_at REAL,
                    created_at REAL
                );
CREATE TABLE IF NOT EXISTS sqlite_sequence(name,seq);
DELETE FROM sqlite_sequence;
INSERT INTO sqlite_sequence VALUES('messages',4);
CREATE TRIGGER msg_fts_insert
                AFTER INSERT ON messages BEGIN
                INSERT INTO messages_fts(rowid, content)
                    VALUES (new.store_id, new.content);
            END;
CREATE TRIGGER msg_fts_delete
                AFTER DELETE ON messages BEGIN
                INSERT INTO messages_fts(messages_fts, rowid, content)
                    VALUES('delete', old.store_id, old.content);
            END;
CREATE TRIGGER msg_fts_update
                AFTER UPDATE OF content ON messages BEGIN
                INSERT INTO messages_fts(messages_fts, rowid, content)
                    VALUES('delete', old.store_id, old.content);
                INSERT INTO messages_fts(rowid, content)
                    VALUES (new.store_id, new.content);
            END;
CREATE TRIGGER msg_fts_stem_insert AFTER INSERT ON messages BEGIN
                INSERT INTO messages_fts_stem(rowid, content) VALUES(new.store_id, new.content);
            END;
CREATE TRIGGER msg_fts_stem_delete AFTER DELETE ON messages BEGIN
                INSERT INTO messages_fts_stem(messages_fts_stem, rowid, content)
                SELECT 'delete', old.store_id, old.content
                WHERE EXISTS (SELECT 1 FROM messages_fts_stem_docsize WHERE id=old.store_id);
            END;
CREATE TRIGGER msg_fts_stem_update AFTER UPDATE OF content ON messages BEGIN
                INSERT INTO messages_fts_stem(messages_fts_stem, rowid, content)
                SELECT 'delete', old.store_id, old.content
                WHERE EXISTS (SELECT 1 FROM messages_fts_stem_docsize WHERE id=old.store_id);
                INSERT INTO messages_fts_stem(rowid, content) VALUES(new.store_id, new.content);
            END;
CREATE TRIGGER nodes_fts_insert
                AFTER INSERT ON summary_nodes BEGIN
                INSERT INTO nodes_fts(rowid, summary)
                    VALUES (new.node_id, new.summary);
            END;
CREATE TRIGGER nodes_fts_delete
                AFTER DELETE ON summary_nodes BEGIN
                INSERT INTO nodes_fts(nodes_fts, rowid, summary)
                    VALUES('delete', old.node_id, old.summary);
            END;
CREATE TRIGGER lcm_rollup_node_insert
            AFTER INSERT ON summary_nodes BEGIN
                INSERT INTO lcm_rollup_invalidations(
                    node_id, scope, covered_start, covered_end, operation
                ) VALUES(
                    new.node_id, new.session_id,
                    MIN(COALESCE(new.earliest_at, new.created_at),
                        COALESCE(new.latest_at, new.created_at)),
                    MAX(COALESCE(new.earliest_at, new.created_at),
                        COALESCE(new.latest_at, new.created_at)), 'insert'
                );
            END;
CREATE TRIGGER lcm_rollup_node_delete
            BEFORE DELETE ON summary_nodes BEGIN
                INSERT INTO lcm_rollup_invalidations(
                    node_id, scope, covered_start, covered_end, operation
                ) VALUES(
                    old.node_id, old.session_id,
                    MIN(COALESCE(old.earliest_at, old.created_at),
                        COALESCE(old.latest_at, old.created_at)),
                    MAX(COALESCE(old.earliest_at, old.created_at),
                        COALESCE(old.latest_at, old.created_at)), 'delete'
                );
            END;
CREATE TRIGGER lcm_rollup_node_update
            AFTER UPDATE OF session_id, depth, summary, token_count,
                            source_token_count, source_ids, source_type, created_at,
                            earliest_at, latest_at, expand_hint ON summary_nodes BEGIN
                INSERT INTO lcm_rollup_invalidations(
                    node_id, scope, covered_start, covered_end, operation
                ) VALUES(
                    old.node_id, old.session_id,
                    MIN(COALESCE(old.earliest_at, old.created_at),
                        COALESCE(old.latest_at, old.created_at)),
                    MAX(COALESCE(old.earliest_at, old.created_at),
                        COALESCE(old.latest_at, old.created_at)), 'update'
                );
                INSERT INTO lcm_rollup_invalidations(
                    node_id, scope, covered_start, covered_end, operation
                ) VALUES(
                    new.node_id, new.session_id,
                    MIN(COALESCE(new.earliest_at, new.created_at),
                        COALESCE(new.latest_at, new.created_at)),
                    MAX(COALESCE(new.earliest_at, new.created_at),
                        COALESCE(new.latest_at, new.created_at)), 'update'
                );
            END;
CREATE TRIGGER lcm_chunk_gapfill_delete
        AFTER DELETE ON lcm_chunk_meta BEGIN
            DELETE FROM metadata WHERE key IN (
                'chunk_gapfill_scan:' || OLD.identity_hash,
                'chunk_policy_revision_through:' || OLD.identity_hash);
        END;
CREATE TRIGGER lcm_chunk_gapfill_update
        AFTER UPDATE OF store_id, identity_hash, chunk_id, char_start, char_end ON lcm_chunk_meta BEGIN
            DELETE FROM metadata WHERE key IN (
                'chunk_gapfill_scan:' || OLD.identity_hash,
                'chunk_policy_revision_through:' || OLD.identity_hash,
                'chunk_gapfill_scan:' || NEW.identity_hash,
                'chunk_policy_revision_through:' || NEW.identity_hash);
        END;
CREATE TRIGGER lcm_chunk_gapfill_message
        AFTER UPDATE OF role, content ON messages
        WHEN lower(trim(OLD.role)) IN ('user', 'assistant')
          OR lower(trim(NEW.role)) IN ('user', 'assistant')
          OR (OLD.role IS NOT NEW.role AND
              (lower(trim(OLD.role)) = 'tool' OR lower(trim(NEW.role)) = 'tool'))
          OR (lower(trim(OLD.role)) = 'tool' AND lower(trim(NEW.role)) = 'tool'
              AND (trim(COALESCE(OLD.content, ''), char(9,10,11,12,13,32)) = '')
                  != (trim(COALESCE(NEW.content, ''), char(9,10,11,12,13,32)) = ''))
        BEGIN
            DELETE FROM metadata WHERE key LIKE 'chunk_gapfill_scan:%'
                OR key LIKE 'chunk_policy_revision_through:%';
        END;
CREATE INDEX idx_msg_session
                ON messages(session_id, store_id);
CREATE INDEX idx_msg_session_ts
                ON messages(session_id, timestamp);
CREATE INDEX idx_lcm_lifecycle_current_session ON lcm_lifecycle_state(current_session_id);
CREATE INDEX idx_lcm_lifecycle_last_finalized_session ON lcm_lifecycle_state(last_finalized_session_id);
CREATE INDEX idx_msg_conversation_session ON messages(conversation_id, session_id, store_id);
CREATE INDEX idx_msg_source_session ON messages(source, session_id, store_id);
CREATE INDEX idx_nodes_session_depth
                ON summary_nodes(session_id, depth, created_at);
CREATE INDEX idx_nodes_session_node
                ON summary_nodes(session_id, node_id);
CREATE INDEX idx_nodes_session_depth_node
                ON summary_nodes(session_id, depth, node_id);
CREATE INDEX idx_nodes_session_latest ON summary_nodes(session_id, latest_at, created_at);
CREATE INDEX idx_lcm_rollup_sources_node
            ON lcm_rollup_sources(node_id);
CREATE INDEX idx_lcm_rollup_invalidations_pending
            ON lcm_rollup_invalidations(event_id);
CREATE INDEX idx_lcm_rollup_invalidations_scope_event
            ON lcm_rollup_invalidations(scope, event_id);
CREATE INDEX idx_lcm_rollup_invalidations_scope_coverage
            ON lcm_rollup_invalidations(scope, covered_start, covered_end, event_id);
CREATE INDEX idx_lcm_rollups_ready_period ON lcm_rollups(scope, period_kind, period_start DESC) WHERE status = 'ready';
CREATE INDEX idx_lcm_rollups_pending ON lcm_rollups(scope, status, failed_at, period_start) WHERE status IN ('stale', 'failed');
CREATE INDEX idx_lcm_rollups_expired_lease ON lcm_rollups(lease_expires_at, rollup_id) WHERE status = 'building';
CREATE INDEX idx_lcm_rollups_stale_day ON lcm_rollups(scope, period_start) WHERE status = 'stale' AND period_kind = 'day';
CREATE INDEX idx_lcm_rollups_stale_aggregate ON lcm_rollups(scope, period_start, period_kind) WHERE status = 'stale' AND period_kind IN ('week', 'month');
CREATE INDEX idx_lcm_rollups_failed_day ON lcm_rollups(scope, failed_at, period_start) WHERE status = 'failed' AND period_kind = 'day';
CREATE INDEX idx_lcm_rollups_failed_aggregate ON lcm_rollups(scope, failed_at, period_start, period_kind) WHERE status = 'failed' AND period_kind IN ('week', 'month');
CREATE INDEX idx_lcm_embedding_profile_model
            ON lcm_embedding_profile(model_name, provider);
CREATE INDEX idx_lcm_embedding_meta_identity_embedded_at
            ON lcm_embedding_meta(identity_hash, embedded_at DESC)
            WHERE archived = 0;
CREATE INDEX idx_lcm_chunk_meta_identity_embedded_at
            ON lcm_chunk_meta(identity_hash, embedded_at DESC)
            WHERE archived = 0;
CREATE INDEX idx_lcm_chunk_meta_store
            ON lcm_chunk_meta(store_id);
CREATE INDEX idx_lcm_embedding_inflight_identity_state ON lcm_embedding_backfill_inflight(identity_hash, state, embedded_id);
CREATE INDEX idx_lcm_embedding_inflight_maintenance ON lcm_embedding_backfill_inflight(identity_hash, state, updated_at, embedded_id);
CREATE INDEX idx_msg_conversation_observed ON messages(conversation_id, observed_at);
CREATE INDEX idx_msg_relations_store ON message_relations(store_id, kind);
CREATE INDEX idx_msg_relations_related ON message_relations(related_store_id, kind);
CREATE INDEX idx_msg_relations_observed ON message_relations(kind, observed_at);
PRAGMA writable_schema=OFF;
COMMIT;
