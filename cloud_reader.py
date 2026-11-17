"""Read-only stdio recall merger; run directly with an isolated interpreter."""
from __future__ import annotations

import argparse
import ast
from concurrent.futures import Future, wait
from contextlib import closing
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import stat
import subprocess
import sys
import threading
import time

RULES = 1
FORGOTTEN_QUERY = "SELECT key FROM metadata WHERE key LIKE 'capture:forgotten:%'"
VIEW = re.compile(r'view-([1-9][0-9]*)-([1-9][0-9]*)-[0-9a-f]{16}')
REPLICA_WAIT_S = 0.25
COPY = re.compile(r'[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}')
QUALIFIED = re.compile(r'([0-9a-f]{32}):(.*)', re.DOTALL)
INTEGER_ARGS = {'store_id', 'node_id', 'after_store_id', 'anchor_store_id'}
SCALARS = INTEGER_ARGS | {'session_id', 'externalized_ref', 'conversation_id'}
LISTS = {'seen_refs', 'exclude_session_ids', 'externalized_refs'}
INTEGER_IDS = INTEGER_ARGS | {'next_cursor', 'adjacent_to', 'later_statement_of'}
INTEGER_LISTS = {'duplicate_store_ids', 'seed_ids', 'has_later_statements'}
PAIRS = {'returned_id_ranges'}
STRING_IDS = {'session_id', 'parent_session_id', 'payload_session_id', 'externalized_ref', 'ref', 'conversation_id'}
STRING_LISTS = {'externalized_refs'}
REFERENCES = {'exact_ref'}
REFERENCE_LISTS = {'novel_refs', 'selected_refs'}
HINTS = {'expand_hint', 'externalized_expand_hint', 'session_hint'}
STORED = {'content', 'content_preview', 'snippet', 'summary', 'tool_calls', 'transcript_content'}
PLAIN = set('''active_session_only best_rank byte_position candidate_files char_budget char_end char_start chunk
    complete content_bytes content_bytes_per_file content_chars content_offset content_returned_chars
    content_scanned_bytes content_scope content_source content_truncated count coverage created_at
    degraded degraded_to_fts delivered delivery_origin depth detail discovery_files discovery_limit
    discovery_truncated displaced diversity_dropped_count earliest_at embedding_query_calls
    embedding_query_tokens embedding_query_tokens_complete escalation_level estimator event_time
    event_time_source expanded_hit_count expanded_hit_limit field_path file_limit found
    from_current_session fts fts_index has_more hydration_policy include kind latest_at limit line
    matched_files max_content_chars message_count mode model ms next_content_offset next_source_offset
    novel_ref_count num_sources omission_reason ordering original_content_bytes original_content_chars
    per_hit_char_cap per_session_limit progress protocol query query_truncated reference_policy
    reference_strict rejected_invalid_or_unreadable rejected_session_mismatch rejected_symlink
    remaining_sources requested requested_seed_limit rerank response_char_cap response_char_limit
    response_chars response_policy response_tokens response_truncated returned_messages returned_sources
    role scan_truncated scanned_files scope_bias score searched seed_limit seen_ref_count session_count
    session_scope sessions_delivered sessions_in_pool snippet_char_cap sort source source_index
    source_limit source_offset source_token_count source_type store_message_count summary_truncated
    termination_reason timestamp token_budget token_count tool_call_id tool_name total_messages
    total_results total_source_tokens total_sources total_tokens truncated type unit
    time_uncertain observed_at ingested_at date stored_at occurred_at event_at event_date precision
    unreferenced_dropped_count unreferenced_omitted_count partial payloads_searched payloads_total prefix_bytes'''.split())
CONTAINERS = set('''adjacent_reply answer_ready arm_weights arms arms_run children coverage delivery delta depths
    embedding_queries evidence_span expanded externalized externalized_payloads externalized_scan hits
    include later_statements messages metrics nodes pagination provenance results roles seed_leads
    sessions snippet_window summary_leads occurrence_time observation_time partial reasons'''.split())
DROPPED = {'store_uuid', 'reader', 'error', 'degraded_reason', 'message_search_error',
           'externalized_note', 'rerank_scores'}
CLASSIFIED = (INTEGER_IDS | INTEGER_LISTS | PAIRS | STRING_IDS | STRING_LISTS | REFERENCES
              | REFERENCE_LISTS | HINTS | STORED | PLAIN | CONTAINERS | {'lineage'})
UNSAFE = re.compile(r'\b(?:store_id|node_id|after_store_id|anchor_store_id)=\d|(?<![0-9a-f]:)\blcm:\d+:|(?<![0-9a-f]:)\blcm-summary:\d')
PLACEHOLDER = re.compile(r"\[(?:(?:Externalized|GC'd externalized) (?:tool output|payload)|Externalized LCM ingest payload):.*?;\s*ref=([^;\]\s]+)\]")
REFERENCE = re.compile(r'lcm:\d+:\d+-\d+|lcm-summary:\d+')
TOOLS = {'lcm_grep', 'lcm_recall', 'lcm_expand', 'lcm_load_session', 'lcm_describe'}


def qualify(prefix, value):
    return f'{prefix}:{value}'


def split_id(value):
    return QUALIFIED.fullmatch(value) if isinstance(value, str) else None


def strip_other_stores(text):
    prefix = '{"other_stores": '
    if not text.startswith(prefix):
        return text
    _, end = json.JSONDecoder().raw_decode(text, len(prefix))
    return '{}' if text[end:] == '}' else '{' + text[end + 2:]


def hint(text, prefix):
    try:
        call = ast.parse(text, mode='eval').body
        if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name):
            return None
        if call.func.id not in TOOLS or call.args:
            return None
        args = {}
        for keyword in call.keywords:
            if keyword.arg is None or keyword.arg in args:
                return None
            value = ast.literal_eval(keyword.value)
            if keyword.arg in SCALARS:
                if type(value) not in (str, int):
                    return None
                value = qualify(prefix, value)
            elif keyword.arg in LISTS:
                if not isinstance(value, list) or any(type(v) not in (int, str) for v in value):
                    return None
                value = [qualify(prefix, v) for v in value]
            args[keyword.arg] = value
        return call.func.id + '(' + ', '.join(f'{k}={json.dumps(v)}' for k, v in args.items()) + ')'
    except (ValueError, SyntaxError, TypeError):
        return None


def rebuild(value, copy_id, result_context=False):
    """Allow-list replica data; stored data is opaque, including nested tool calls."""
    if isinstance(value, list):
        return [result for item in value if (not result_context or isinstance(item, dict)) and (result := rebuild(item, copy_id, result_context)) is not None]
    if not isinstance(value, dict):
        return value if not isinstance(value, str) or (len(value) <= 512 and not UNSAFE.search(value)) else None
    if 'message_search_error' in value:
        raise ValueError('replica search failed')
    out = {}
    for key, item in value.items():
        result = None
        if key in STORED or (key in PLAIN and item is None):
            out[key] = copy.deepcopy(item)
            continue
        if key in INTEGER_IDS and type(item) is int:
            result = qualify(copy_id, item)
        elif key in INTEGER_LISTS and isinstance(item, list) and all(type(v) is int for v in item):
            result = [qualify(copy_id, v) for v in item]
        elif key in PAIRS and isinstance(item, list) and all(isinstance(p, list) and len(p) == 2 and all(type(v) is int for v in p) for p in item):
            result = [[qualify(copy_id, v) for v in pair] for pair in item]
        elif key in STRING_IDS and isinstance(item, str):
            result = qualify(copy_id, item)
        elif key in STRING_LISTS and isinstance(item, list) and all(isinstance(v, str) for v in item):
            result = [qualify(copy_id, v) for v in item]
        elif key in REFERENCES and isinstance(item, str) and REFERENCE.fullmatch(item):
            result = qualify(copy_id, item)
        elif key in REFERENCE_LISTS and isinstance(item, list):
            result = [qualify(copy_id, v) for v in item if isinstance(v, str) and REFERENCE.fullmatch(v)]
        elif key in HINTS and isinstance(item, str):
            result = hint(item, copy_id)
        elif key == 'lineage' and isinstance(item, str) and re.fullmatch(r'subagent of [^\s:]+', item):
            result = 'subagent of ' + qualify(copy_id, item[len('subagent of '):])
        elif key in PLAIN and (item is None or type(item) in (bool, int, float, str)):
            result = item if not isinstance(item, str) or (len(item) <= 512 and not UNSAFE.search(item) and not split_id(item)) else None
        elif (key in CONTAINERS or re.fullmatch(r'd\d+', key)) and isinstance(item, (dict, list)):
            result = rebuild(item, copy_id, key in {'results', 'hits', 'messages', 'expanded', 'children', 'nodes', 'sessions', 'summary_leads', 'seed_leads'})
        if result is not None:
            out[key] = result
    refs = list(out.get('externalized_refs', []))
    for key in STORED & value.keys():
        for match in PLACEHOLDER.finditer(json.dumps(value[key], ensure_ascii=False) if not isinstance(value[key], str) else value[key]):
            ref = qualify(copy_id, match[1])
            if ref not in refs:
                refs.append(ref)
    if refs:
        out['externalized_refs'] = refs
    return out


def error_body(text, args, prefix):
    for key, value in args.items():
        values = value if key in LISTS and isinstance(value, list) else [value] if key in SCALARS else []
        for item in values:
            if type(item) in (int, str) and str(item):
                text = replace_token(text, item, qualify(prefix, item))
    if UNSAFE.search(text) or any(v != prefix for v in re.findall(r'[0-9a-f]{32}', text)):
        return {'error': 'the other store could not answer this call', 'store': prefix}
    return {'error': text}


def replace_token(text, old, new):
    return re.sub(r'(?<![\w:.-])' + re.escape(str(old)) + r'(?![\w.-])', lambda _: str(new), text)


def reader_args(argv, folder):
    argv = list(argv)
    for flag, value in (('--store', folder / 'lcm.db'), ('--hermes-home', folder)):
        argv[argv.index(flag) + 1] = str(value)
    return argv


def forgotten(database):
    with database.open('rb') as handle:
        handle.seek(18)
        header = handle.read(2)
    if len(header) != 2 or (max(header) >= 2 and not all(Path(str(database) + suffix).is_file() for suffix in ('-wal', '-shm'))):
        raise ValueError('unsafe local database')
    with closing(sqlite3.connect(database.resolve().as_uri() + '?mode=ro', uri=True)) as conn:
        ids = {row[0][len('capture:forgotten:'):] for row in conn.execute(FORGOTTEN_QUERY)}
    return ids, forgotten_digest(ids)


def forgotten_digest(ids):
    return hashlib.sha256(b''.join(x + b'\n' for x in sorted({s.encode('utf-8') for s in ids}))).hexdigest()


def removed_sha256(copy_folder):
    try:
        raw = (copy_folder / 'removed.json').read_bytes()
    except FileNotFoundError:
        raw = b''
    return hashlib.sha256(raw).hexdigest()


class Child:
    def __init__(self, argv, budget, folder=None):
        self.fd = None
        self.process = None
        self.guard = threading.Lock()
        self.busy = threading.Lock()
        self.budget = budget
        self.folder = folder
        try:
            if folder:
                self.fd = os.open(folder, os.O_RDONLY | os.O_NOFOLLOW)
                fcntl.flock(self.fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
                mode, held = folder.lstat(), os.fstat(self.fd)
                if not stat.S_ISDIR(mode.st_mode) or (mode.st_dev, mode.st_ino) != (held.st_dev, held.st_ino):
                    raise ValueError('bad view folder')
            self.process = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                            stderr=subprocess.DEVNULL, close_fds=True)
        except Exception:
            self.close()
            raise

    def close(self):
        with self.guard:
            try:
                if self.process:
                    try:
                        if self.process.poll() is None:
                            self.process.kill()
                    except ProcessLookupError:
                        pass
                    self.process.wait()
                    for stream in (self.process.stdin, self.process.stdout):
                        try:
                            stream.close()
                        except OSError:
                            pass
            finally:
                if self.fd is not None:
                    try:
                        os.close(self.fd)
                    except OSError:
                        pass
                    self.fd = None

    def ask(self, raw, notification=False):
        future = Future()
        if not self.busy.acquire(blocking=False):
            return None
        def run():
            timer = threading.Timer(self.budget, self.close) if self.budget is not None else None
            if timer:
                timer.daemon = True
                timer.start()
            try:
                self.process.stdin.write(raw)
                self.process.stdin.flush()
                reply = b'' if notification else self.process.stdout.readline()
                if not notification and not reply:
                    raise RuntimeError('reader unavailable')
                result, error = reply, None
            except Exception as exc:
                result, error = None, exc
                try:
                    self.close()
                except Exception:
                    pass
            finally:
                if timer:
                    timer.cancel()
                self.busy.release()
            if error:
                future.set_exception(error)
            else:
                future.set_result(result)
        threading.Thread(target=run, daemon=True).start()
        return future


def request(ident, method, params=None):
    return (json.dumps({'jsonrpc': '2.0', 'id': ident, 'method': method, 'params': params or {}}) + '\n').encode()


def tool_request(name, args):
    return request(1, 'tools/call', {'name': name, 'arguments': args})


def tool_body(raw):
    reply = json.loads(raw)
    return json.loads(reply['result']['content'][0]['text']), reply


class Replica:
    def __init__(self, prefix, argv, budget):
        self.prefix, self.argv, self.budget = prefix, argv, budget
        self.child = self.pending = self.probe = None
        self.generation = self.target = None
        self.status = 'unavailable'
        self.digests = self.target_digests = self.target_folder = None

    def close(self):
        for child in (self.child, self.pending):
            if child:
                child.close()

    def refresh(self, folder, generation, digests):
        if self.child and self.digests != digests:
            self.child.close()
            self.child = None
        if self.pending and (self.target_folder != folder or self.target_digests != digests):
            self.pending.close()
            self.pending = self.probe = None
        if self.child and self.child.process.poll() is not None:
            self.child.close()
            self.child = None
            self.status = 'unavailable'
            return  # Expose one unavailable call before restarting.
        if self.probe and self.probe.done():
            if self.child and self.child.busy.locked():
                self.status = 'ok'
                return
            try:
                self.probe.result()
                old, self.child = self.child, self.pending
                self.generation, self.digests = self.target, self.target_digests
                if old:
                    old.close()  # Main dispatch is serial; no old call is in flight here.
                self.status = 'ok'
            except Exception:
                self.pending.close()
                self.status = 'ok' if self.child else 'unavailable'
                self.pending = self.probe = None
                return
            self.pending = self.probe = None
        if self.child and self.child.folder == folder:
            self.status = 'ok'
            return
        if self.pending:
            self.status = 'ok' if self.child else 'warming'
            return
        self.pending = Child(reader_args(self.argv, folder), self.budget, folder)
        self.target, self.target_folder, self.target_digests = generation, folder, digests
        self.probe = Future()
        child, probe = self.pending, self.probe
        def warm():
            try:
                child.ask(request(0, 'initialize', {'protocolVersion': '2025-06-18'})).result()
                body, reply = tool_body(child.ask(tool_request('lcm_grep', {'query': 'lcmx_probe_' + os.urandom(16).hex()})).result())
                if reply['result'].get('isError') or 'error' in body:
                    raise ValueError('probe refused')
                probe.set_result(None)
            except Exception as exc:
                try:
                    child.close()
                finally:
                    probe.set_exception(exc)
        threading.Thread(target=warm, daemon=True).start()
        self.status = 'ok' if self.child else 'warming'


class Merger:
    def __init__(self, argv, replica_wait=REPLICA_WAIT_S, budget=8):
        self.argv, self.replica_wait, self.budget = argv, replica_wait, budget
        self.home = Path(argv[argv.index('--hermes-home') + 1])
        self.database = Path(argv[argv.index('--store') + 1])
        self.local = None
        self.enabled_tools = None
        self.replicas = {}
        self.builders = {}
        self.own = None
        path = self.home / 'cloud' / 'cloud.json'
        try:
            mode = path.lstat()
            if stat.S_ISREG(mode.st_mode) and mode.st_uid == os.getuid() and not mode.st_mode & 0o022:
                own = json.loads(path.read_text()).get('copy_id')
                if isinstance(own, str) and COPY.fullmatch(own):
                    self.own = own.replace('-', '')
        except (OSError, ValueError, AttributeError):
            pass
        try:
            self.local = Child(argv, None)
            if self.own:
                try:
                    tools = json.loads(self.local_call(request('lcmx-private-tools', 'tools/list')))['result']['tools']
                    self.enabled_tools = {tool['name'] for tool in tools}
                except Exception:
                    pass
                self.refresh()
        except Exception:
            self.close()
            raise

    def close(self):
        for reader in [self.local, *self.replicas.values()]:
            if reader:
                try:
                    reader.close()
                except OSError:
                    pass

    def start_build(self, folder, digests):
        child = self.builders.get(folder.name)
        if child and child.poll() is None:
            return
        self.builders.pop(folder.name, None)
        try:
            current = (folder / 'current').read_bytes()
            generation = int(current) if re.fullmatch(rb'[1-9][0-9]*\n?', current) else None
            result = json.loads((folder / 'view-result.json').read_bytes())
            same = (result.get('generation') == generation and result.get('rules') == RULES
                    and (result.get('forgotten_sha256'), result.get('removed_sha256')) == digests)
            if same and result.get('code') in {'view_unsupported_table', 'view_unsupported_key', 'view_summary_spans_forgotten'}:
                return
            if same and result.get('code') in {'view_busy', 'view_disk_floor'} and 0 <= time.time() - result['ended_at'] < 60:
                return
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            pass
        child = subprocess.Popen([sys.executable, '-I', '-S', '-B', str(Path(__file__).parent / 'scripts/lcm_x_cloud.py'),
                                  'view', '--home', str(self.home), '--copy-id', folder.name],
                                 stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                 stderr=subprocess.DEVNULL, start_new_session=True)
        self.builders[folder.name] = child
        try:
            os.setpriority(os.PRIO_PROCESS, child.pid, 19)
        except (AttributeError, OSError):
            pass

    def refresh(self, digest=None):
        for replica in self.replicas.values():
            replica.status = 'unavailable'
        if digest is None:
            try:
                _, digest = forgotten(self.database)
            except Exception:
                return
        root = self.home / 'cloud' / 'replicas'
        if not root.exists():
            return
        for copy_folder in sorted(root.iterdir()):
            if not COPY.fullmatch(copy_folder.name) or copy_folder.is_symlink() or not copy_folder.is_dir():
                continue
            prefix = copy_folder.name.replace('-', '')
            if prefix == self.own:
                continue
            replica = self.replicas.setdefault(prefix, Replica(prefix, self.argv, self.budget))
            builder = self.builders.get(copy_folder.name)
            if builder and builder.poll() is not None:
                self.builders.pop(copy_folder.name)
            digests = None
            try:
                digests = digest, removed_sha256(copy_folder)
                pointer = json.loads((copy_folder / 'view.json').read_bytes())
                name = pointer['folder']
                match = VIEW.fullmatch(name)
                if (type(pointer.get('format')) is not int or pointer['format'] != 1 or not match or type(pointer.get('generation')) is not int
                        or int(match[1]) != pointer['generation'] or int(match[2]) != RULES
                        or type(pointer.get('rules')) is not int or pointer['rules'] != RULES
                        or (pointer.get('forgotten_sha256'), pointer.get('removed_sha256')) != digests):
                    raise ValueError('stale view')
                if self.enabled_tools is not None:
                    replica.refresh(copy_folder / name, pointer['generation'], digests)
            except Exception:
                replica.close()
                replica.child = replica.pending = replica.probe = None
                replica.status = 'warming'
                if digests is not None:
                    try:
                        self.start_build(copy_folder, digests)
                    except Exception:
                        pass  # A failed builder launch cannot affect the local answer.

    def local_call(self, raw, notification=False):
        for attempt in range(2):
            try:
                return self.local.ask(raw, notification).result()
            except Exception:
                self.local.close()
                if attempt:
                    raise
                self.local = Child(self.argv, None)
                self.enabled_tools = None

    def route(self, args):
        values = [v for k, v in args.items() if k in SCALARS]
        values += args.get('externalized_refs', []) if isinstance(args.get('externalized_refs'), list) else []
        prefixes = {m[1] for v in values if (m := split_id(v))}
        if len(prefixes) > 1:
            raise ValueError('ids from more than one store in one call')
        prefix = next(iter(prefixes), None)
        if prefix and prefix != self.own and prefix not in self.replicas:
            name = '-'.join((prefix[:8], prefix[8:12], prefix[12:16], prefix[16:20], prefix[20:]))
            folder = self.home / 'cloud' / 'replicas' / name
            if folder.is_dir() and not folder.is_symlink():
                self.replicas[prefix] = Replica(prefix, self.argv, self.budget)
        return prefix if prefix in self.replicas or prefix == self.own else None

    def arguments(self, args, prefix, strip=False):
        out = copy.deepcopy(args)
        local = prefix in (None, self.own)
        for key, value in args.items():
            if key in SCALARS and strip and (match := split_id(value)) and match[1] in (prefix, self.own):
                bare = match[2]
                out[key] = int(bare) if key in INTEGER_ARGS and re.fullmatch(r'-?\d+', bare) else bare
            if key == 'externalized_refs' and isinstance(value, list):
                out[key] = [m[2] if strip and (m := split_id(v)) and m[1] == prefix else v for v in value]
                continue
            if key in LISTS and isinstance(value, list):
                out[key] = []
                for item in value:
                    match = split_id(item)
                    if match and match[1] == self.own:
                        if local:
                            out[key].append(match[2])
                    elif match and match[1] in self.replicas:
                        if match[1] == prefix:
                            out[key].append(match[2])
                    elif local or (not match and key == 'exclude_session_ids'):
                        out[key].append(item)
        return out

    def replica_call(self, replica, name, args):
        if replica.status != 'ok' or not replica.child:
            return None
        return replica.child.ask(tool_request(name, args))

    def rebuilt(self, raw, replica, args):
        body, _ = tool_body(raw)
        if 'message_search_error' in body:
            raise ValueError('replica search failed')
        if isinstance(body.get('error'), str):
            return error_body(body['error'], args, replica.prefix)
        return rebuild(body, replica.prefix)

    def routed(self, replica, name, args):
        if replica.status != 'ok' or not replica.child:
            return {'error': 'the other store could not answer this call', 'store': replica.prefix}
        future = self.replica_call(replica, name, args)
        if future is None:
            return {'error': 'the other store could not answer this call', 'store': replica.prefix}
        body = self.rebuilt(future.result(), replica, args)
        return {'store': replica.prefix, 'generation': replica.generation, **body}

    def handle(self, raw):
        try:
            req = json.loads(raw)
        except ValueError:
            return self.local_call(raw)
        if not isinstance(req, dict):
            return self.local_call(raw)
        notification = 'id' not in req and isinstance(req.get('method'), str)
        if not self.own or notification:
            return self.local_call(raw, notification)
        method = req.get('method')
        if method == 'tools/list':
            reply = json.loads(self.local_call(raw))
            for tool in reply['result']['tools']:
                for key, schema in tool['inputSchema'].get('properties', {}).items():
                    if key in SCALARS | LISTS:
                        if schema.get('type') == 'integer':
                            schema['type'] = ['integer', 'string']
                        schema['description'] = schema.get('description', '') + ' Ids from another store carry a <store>: prefix and must be passed back unchanged.'
            return (json.dumps(reply, ensure_ascii=False) + '\n').encode()
        if method != 'tools/call':
            return self.local_call(raw)
        params = req.get('params') or {}
        if not isinstance(params, dict):
            return self.local_call(raw)
        name, args = params.get('name'), params.get('arguments') or {}
        if not isinstance(args, dict):
            return self.local_call(raw)
        if self.enabled_tools is None:
            try:
                tools = json.loads(self.local_call(request('lcmx-private-tools', 'tools/list')))['result']['tools']
                self.enabled_tools = {tool['name'] for tool in tools}
            except Exception:
                return self.local_call(raw)
        if name not in self.enabled_tools:
            return self.local_call(raw)
        try:
            prefix = self.route(args)
        except ValueError as exc:
            return self.answer(req, {'error': str(exc)}, True)
        try:
            hidden, digest = forgotten(self.database)
            self.refresh(digest)
        except Exception:
            hidden = None
            for replica in self.replicas.values():
                replica.status = 'unavailable'
        if prefix and prefix != self.own:
            replica = self.replicas[prefix]
            try:
                body = self.routed(replica, name, self.arguments(args, prefix, True)) if hidden is not None else None
            except Exception:
                body = None
            return self.answer(req, body or {'error': 'the other store could not answer this call', 'store': prefix}, bool(not body or 'error' in body))
        local_args = self.arguments(args, self.own if prefix == self.own else None, prefix == self.own)
        local_raw = raw if local_args == args else request(req['id'], method, {**params, 'arguments': local_args})
        if name not in ('lcm_grep', 'lcm_recall') or not self.replicas:
            return self.local_call(local_raw)
        pending, entries = {}, []
        for replica in self.replicas.values():
            skip = bool(args.get('externalized_refs'))
            entry = {'copy_id': replica.prefix, 'generation': replica.generation, 'status': 'ok' if skip else replica.status, 'hits': []}
            entries.append(entry)
            if hidden is None:
                entry['status'] = 'unavailable'
            elif not skip:
                replica_args = self.arguments(args, replica.prefix, True)
                future = self.replica_call(replica, name, replica_args)
                if future:
                    pending[future] = (replica, entry)
                elif replica.status == 'ok':
                    entry['status'] = 'late'
        reply_raw = self.local_call(local_raw)
        if pending:
            wait(pending, timeout=self.replica_wait)
        for future, (replica, entry) in pending.items():
            if not future.done():
                entry['status'] = 'late'
                continue
            try:
                body = self.rebuilt(future.result(), replica, args)
                if body is None or 'error' in body:
                    raise ValueError('replica refused')
                if 'partial' in body:
                    entry['partial'] = body['partial']
                hits = body.get('results' if name == 'lcm_grep' else 'hits', [])
                entry['hits'] = hits if isinstance(hits, list) else []
                sessions = body.get('provenance', {}).get('delivery', {}).get('sessions')
                if isinstance(sessions, list):
                    entry['sessions'] = sessions
            except Exception:
                entry['status'] = 'unavailable'
        reply = json.loads(reply_raw)
        content = reply['result']['content'][0]
        text = content['text']
        if not text.startswith('{'):
            return reply_raw
        prefix_text = json.dumps({'other_stores': sorted(entries, key=lambda e: e['copy_id'])})[:-1]
        content['text'] = prefix_text + ('}' if text == '{}' else ', ' + text[1:])
        return (json.dumps(reply, ensure_ascii=False) + '\n').encode()

    @staticmethod
    def answer(req, body, error=False):
        result = {'content': [{'type': 'text', 'text': json.dumps(body, ensure_ascii=False)}], 'isError': error}
        return (json.dumps({'jsonrpc': '2.0', 'id': req['id'], 'result': result}, ensure_ascii=False) + '\n').encode()


def main():
    try:
        argv = json.loads(os.environ['LCMX_LOCAL_READER_CMD'])
        if not isinstance(argv, list) or not argv or not all(isinstance(v, str) for v in argv):
            raise ValueError('invalid argv')
    except (KeyError, ValueError, TypeError):
        sys.stderr.write('LCMX_LOCAL_READER_CMD must be a JSON array of strings\n')
        return 2
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replica-wait-s', type=float, default=REPLICA_WAIT_S)
    parser.add_argument('--budget-s', type=float, default=8)
    merger = None
    try:
        try:
            options = parser.parse_args()
        except SystemExit as exc:
            if exc.code:
                os.execvp(argv[0], argv)
            return 0
        merger = Merger(argv, max(0, options.replica_wait_s), max(0.01, options.budget_s))
    except Exception:
        if merger:
            merger.close()
        os.execvp(argv[0], argv)
    try:
        for raw in sys.stdin.buffer:
            if raw.strip():
                try:
                    reply = merger.handle(raw)
                except Exception:
                    try:
                        req = json.loads(raw)
                    except Exception:
                        req = None
                    notification = isinstance(req, dict) and 'id' not in req and isinstance(req.get('method'), str)
                    try:
                        reply = merger.local_call(raw, notification)
                    except Exception:
                        reply = b'' if notification else (json.dumps({'jsonrpc': '2.0', 'id': req.get('id') if isinstance(req, dict) else None,
                                             'error': {'code': -32603, 'message': 'Internal error'}}) + '\n').encode()
                if reply:
                    sys.stdout.buffer.write(reply)
                    sys.stdout.buffer.flush()
        return 0
    except Exception:
        return 1
    finally:
        merger.close()


if __name__ == '__main__':
    raise SystemExit(main())
