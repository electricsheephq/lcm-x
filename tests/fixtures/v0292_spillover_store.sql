-- #2521 fixture: a FIRST-PHASE store written by the real v0.29.2 engine (commit 261d41ea), before any upgrade.
-- Shape: session u1-session, conversation u1-conv; stamped user, assistant tool call, tool result = a host spill-over
-- marker stored bare (v0.29.2 cannot recover cache/spillover), assistant reply. The home path inside the marker text
-- (messages table only) is replaced by @HERMES_HOME@; the test substitutes its temp home and writes the spill file.
-- The messages_fts and messages_fts_stem shadow rows are rebuilt over the placeholder text (no machine path); the
-- test rebuilds both indexes after substituting its home, as v0.29.2 would have indexed that path.
-- Regenerate: drive the v0.29.2 engine (on_session_start, ingest, shutdown) with the same history, then sqlite3 .dump.
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
INSERT INTO messages VALUES(1,'u1-session','cli','u1-conv','user','Please dump the big report.',NULL,NULL,NULL,1794266036.453254,11,0,1794266036.453254,1762700001.0,'host_message_timestamp',NULL,NULL);
INSERT INTO messages VALUES(2,'u1-session','cli','u1-conv','assistant','',NULL,'[{"id": "call_spill", "type": "function", "function": {"name": "dump", "arguments": "{}"}}]',NULL,1794266036.4533529,10,0,1794266036.4533529,1762700002.0,'host_message_timestamp',NULL,NULL);
INSERT INTO messages VALUES(3,'u1-session','cli','u1-conv','tool','<persisted-output>
This tool result was too large (5,427 characters, 5.3 KB).
Full output saved to: @HERMES_HOME@/cache/spillover/call_spill.txt
Use the read_file tool with offset and limit to access specific sections of this output.
Recovery: page through the saved file with read_file (offset/limit) or process it with execute_code — do NOT re-request the same data from the remote API; the full result is already on disk.

Preview (first 30 chars):
SPILLOVER_RECOVERY_NEEDLE:
é文a
...
</persisted-output>','call_spill',NULL,NULL,1794266036.453408,144,0,1794266036.453408,1762700003.0,'host_message_timestamp',NULL,NULL);
INSERT INTO messages VALUES(4,'u1-session','cli','u1-conv','assistant','The report is saved; the needle is at the top.',NULL,NULL,NULL,1794266036.4534369,16,0,1794266036.4534369,1762700004.0,'host_message_timestamp',NULL,NULL);
CREATE TABLE metadata (
                key TEXT PRIMARY KEY,
                value TEXT
            );
INSERT INTO metadata VALUES('fts_integrity_checked_at:messages_fts','1794266036.391534');
INSERT INTO metadata VALUES('schema_version','5');
INSERT INTO metadata VALUES('fts_stem_state','ready');
INSERT INTO metadata VALUES('fts_integrity_checked_at:nodes_fts','1794266036.394088');
INSERT INTO metadata VALUES('ignored_active_replay_placeholder_hash_counts:u1-session','{}');
INSERT INTO metadata VALUES('ignored_active_replay_placeholder_hash_ordinals:u1-session','{}');
PRAGMA writable_schema=ON;
INSERT INTO sqlite_schema(type,name,tbl_name,rootpage,sql)VALUES('table','messages_fts','messages_fts',0,'CREATE VIRTUAL TABLE "messages_fts" USING fts5(
                        "content",
                        content="messages",
                        content_rowid="store_id"
                    )');
CREATE TABLE IF NOT EXISTS 'messages_fts_data'(id INTEGER PRIMARY KEY, block BLOB);
INSERT INTO messages_fts_data VALUES(1,x'0465');
INSERT INTO messages_fts_data VALUES(10,x'000000000101010001010101');
INSERT INTO messages_fts_data VALUES(137438953473,x'0000026e02303303020e020130030250010334323703020b01013503040a05010661636365737303022502066c726561647903024b02026e640302220202706903024602017404020901036269670102050105636163686503021603026c6c030218020968617261637465727303020c05017303025102036f646503023b010464617461030242020369736b03024d02016f03023c0203756d7001020301076578656375746503023a0204e6968761030255010466696c6503061e1405030372737403024f0203726f6d0302430203756c6c0304103a01066865726d657303021402036f6d650302150102697303024a0104040602017403023801026b6203020f01056c617267650302090204696d69740304231401066e6565646c6503025401020702026f7403023d01026f660302280304667365740304211502016e03024c02017203023602057574707574030803101b2f01047061676503022c020865727369737465640304025602056c65617365010202020672657669657703024e03056f636573730302370102726503023e0302616403041d170306636f7665727903042b2a03046d6f74650302450304706f72740102060302030305717565737403023f030473756c7403040645010473616d6503024103037665640304121f0102050207656374696f6e730302270207706563696669630302260303696c6c03021906046f7665720304173d0103746865010204020a1c14140605010602060603026973030404270305726f75676803022d02016f0304131303016f03020804016c0304051c03017004020b0202787403021a010375736503021b01037761730302070203697468030620130a04060608070b0b070706080a070e0608090806080c090b0808090b080b06070a0a0e07070a06060d090e0a0b0a07080c090c0a0a090c0c0c080a14080a07060706070808');
CREATE TABLE IF NOT EXISTS 'messages_fts_idx'(segid, term, pgno, PRIMARY KEY(segid, term)) WITHOUT ROWID;
INSERT INTO messages_fts_idx VALUES(1,x'',2);
CREATE TABLE IF NOT EXISTS 'messages_fts_docsize'(id INTEGER PRIMARY KEY, sz BLOB);
INSERT INTO messages_fts_docsize VALUES(1,x'05');
INSERT INTO messages_fts_docsize VALUES(2,x'00');
INSERT INTO messages_fts_docsize VALUES(3,x'56');
INSERT INTO messages_fts_docsize VALUES(4,x'0a');
CREATE TABLE IF NOT EXISTS 'messages_fts_config'(k PRIMARY KEY, v) WITHOUT ROWID;
INSERT INTO messages_fts_config VALUES('version',4);
CREATE TABLE lcm_migration_state (
            step_name TEXT PRIMARY KEY,
            completed_at REAL NOT NULL
        );
INSERT INTO lcm_migration_state VALUES('v2_external_content_fts_triggers',1794266036.0);
INSERT INTO lcm_migration_state VALUES('v3_lifecycle_state',1794266036.0);
INSERT INTO lcm_migration_state VALUES('v4_lifecycle_debt_columns',1794266036.0);
INSERT INTO lcm_migration_state VALUES('v5_message_conversation_id',1794266036.0);
INSERT INTO lcm_migration_state VALUES('messages_ingested_at_backfill_v1',1794266036.0);
INSERT INTO lcm_migration_state VALUES('fts_stem_v1',1794266036.0);
INSERT INTO lcm_migration_state VALUES('temporal_rollups_v1',1794266036.0);
INSERT INTO lcm_migration_state VALUES('embeddings_v1',1794266036.0);
INSERT INTO lcm_migration_state VALUES('chunk_vectors_v1',1794266036.0);
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
INSERT INTO lcm_lifecycle_state VALUES('u1-conv','u1-session',NULL,0,0,NULL,0,1794266036.4001222,NULL,NULL,NULL,NULL,NULL,1794266036.4001222);
INSERT INTO sqlite_schema(type,name,tbl_name,rootpage,sql)VALUES('table','messages_fts_stem','messages_fts_stem',0,'CREATE VIRTUAL TABLE messages_fts_stem USING fts5(content, content=''messages'', content_rowid=''store_id'', tokenize=''porter unicode61'')');
CREATE TABLE IF NOT EXISTS 'messages_fts_stem_data'(id INTEGER PRIMARY KEY, block BLOB);
INSERT INTO messages_fts_stem_data VALUES(1,x'0465');
INSERT INTO messages_fts_stem_data VALUES(10,x'000000000101010001010101');
INSERT INTO messages_fts_stem_data VALUES(137438953473,x'0000025702303303020e020130030250010334323703020b01013503040a05010661636365737303022502066c726561646903024b02026e6403022202027069030246020174040209010362696701020501046361636803021603026c6c0302180203686172030251050361637403020c02036f646503023b010464617461030242020369736b03024d02016f03023c0203756d70010203010665786563757403023a0204e6968761030255010466696c6503061e1405030372737403024f0203726f6d0302430203756c6c0304103a01046865726d03021402036f6d650302150102697303024a0104040602017403023801026b6203020f01046c6172670302090204696d69740304231401056e6565646c03025401020702026f7403023d01026f660302280304667365740304211502016e03024c02017203023602057574707574030803101b2f01047061676503022c02066572736973740304025602046c656173010202020672657669657703024e03056f636573730302370102726503023e0302616403041d170306636f7665726903042b2a03036d6f740302450304706f72740102060302030305717565737403023f030473756c7403040645010473616d65030241030276650304121f0102050206656374696f6e030227020570656369660302260303696c6c03021906026f760304173d0103746865010204020a1c141406050106020606030169030404270305726f75676803022d02016f0304131303016f03020804016c0304051c03017004020b0202787403021a0102757303021b010277610302070203697468030620130a04060608070b0b070706080907080808090806080b090b08080909080b0607090a0d07070a06060d090c090b0a07080c080c0a0a090b0b0a080814070a07060706070707');
CREATE TABLE IF NOT EXISTS 'messages_fts_stem_idx'(segid, term, pgno, PRIMARY KEY(segid, term)) WITHOUT ROWID;
INSERT INTO messages_fts_stem_idx VALUES(1,x'',2);
CREATE TABLE IF NOT EXISTS 'messages_fts_stem_docsize'(id INTEGER PRIMARY KEY, sz BLOB);
INSERT INTO messages_fts_stem_docsize VALUES(1,x'05');
INSERT INTO messages_fts_stem_docsize VALUES(2,x'00');
INSERT INTO messages_fts_stem_docsize VALUES(3,x'56');
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
