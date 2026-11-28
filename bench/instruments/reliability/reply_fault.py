"""Held-reply fault: harness-owned reservation, host-row witness, and fail-closed evidence."""
import json
import os
import sqlite3
import threading
import time

KIND = "crash_after_host_reply_before_store"
R1_REASON = "acp-process only: the held-reply crash needs a real host process"


def readonly(path):
    return sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, isolation_level=None, timeout=0)


def invalid_verdict(rec, reason):
    earlier = rec.get("reason") if rec.get("verdict") == "ERROR" else None
    rec.update(verdict="ERROR", reason="reply-held fault INVALID: " + reason +
               ("; earlier: " + earlier[:300] if earlier else ""))


class HeldReply:
    def __init__(self, run, fault, text, timeout=5.0):
        self.run, self.fault, self.text, self.timeout = run, fault, text, timeout
        self.position = fault.get("at", "final_reply")
        self.ids = text if self.position == "tool_results" else []
        self.done, self.reservation, self.host = threading.Event(), None, None
        self.polls = []  # Each poll's wall time and count/busy result; never content or row ids.
        self.record = dict(turn=fault["turn"], phase="A", valid=False, invalid_reason=None,
                           session=None, parent=None, root=None, depth=None, journal_mode=None,
                           reservation_wall=None, reservation_result=None, release_wall=None,
                           first_poll_wall=None, seen_poll_wall=None, kill_wall=None, poll_count=0,
                           busy_poll_count=0, kill_offset_ms=None, exit_confirmed_ms=None, host_has_row=None,
                           store_has_row=None, restart_exit=None, polls=[], position=self.position, witness_rows=0)

    def invalid(self, reason):
        self.record.update(valid=False, invalid_reason=self.record["invalid_reason"] or reason)

    def before_release(self):
        try:
            self.reservation = sqlite3.connect(self.run.home / "lcm.db", timeout=5, isolation_level=None)
            self.record["journal_mode"] = self.reservation.execute("PRAGMA journal_mode").fetchone()[0]
            self.host = readonly(self.run.home / "state.db")
            self.max_id = self.host.execute("SELECT COALESCE(MAX(id), 0) FROM messages").fetchone()[0]
            self.record["reservation_wall"] = time.time()
            self.reservation.execute("BEGIN IMMEDIATE")
            self.record["reservation_result"] = "ok"
        except Exception as exc:
            self.record["reservation_result"] = type(exc).__name__
            self.invalid("reservation: " + type(exc).__name__)
            self.finish()
        self.record["release_wall"] = time.time()

    def release_failed(self):
        self.invalid("response release failed")
        try:
            self.stop_invalid_tool_loop()
        finally:
            self.finish()

    def stop_invalid_tool_loop(self):
        r = self.record
        if self.position == "tool_results" and r["kill_wall"] is None:
            r["kill_wall"], r["kill_monotonic_ns"] = time.time(), time.monotonic_ns()
            self.run.proc.kill()  # Close any held next request, even if reservation/release failed.
            if not self.wait_for_exit():
                self.invalid("host group did not exit")

    def release_reservation(self):
        if self.reservation is not None:
            try:
                if self.reservation.in_transaction:
                    self.reservation.execute("ROLLBACK")
            finally:
                self.reservation.close()
                self.reservation = None

    def finish(self):
        try:
            self.release_reservation()
        finally:
            try:
                if self.host is not None:
                    self.host.close()
                    self.host = None
            finally:
                self.done.set()

    def after_release(self):
        if self.done.is_set() or (self.position == "tool_results" and self.record["invalid_reason"]):
            try:
                self.stop_invalid_tool_loop()
            finally:
                self.finish()
            return
        r, end = self.record, time.monotonic() + self.timeout
        try:
            while time.monotonic() < end:
                wall = time.time()
                r["first_poll_wall"] = r["first_poll_wall"] or wall
                try:
                    if self.ids:
                        rows = self.host.execute("SELECT id, session_id, tool_call_id FROM messages WHERE id > ? AND role = 'tool' "
                                                 f"AND tool_call_id IN ({','.join('?' for _ in self.ids)}) ORDER BY id",
                                                 (self.max_id, *self.ids)).fetchall()
                    else:
                        rows = self.host.execute("SELECT id, session_id FROM messages WHERE id > ? "
                                                 "AND role = 'assistant' AND content = ? ORDER BY id",
                                                 (self.max_id, self.text)).fetchall()
                    self.polls.append((wall, len(rows)))
                except sqlite3.OperationalError:
                    self.polls.append((wall, "busy"))
                    r["busy_poll_count"] += 1
                    rows = []
                r["poll_count"] += 1
                r["polls"].append([round(wall, 4), self.polls[-1][1]])
                if rows and (not self.ids or {row[2] for row in rows} == set(self.ids)):
                    r["seen_poll_wall"], r["kill_wall"] = wall, time.time()
                    r["kill_monotonic_ns"] = time.monotonic_ns()
                    self.run.proc.kill()  # No DB read between the witness and the signal.
                    r["witness_rows"] = len(rows)
                    r["kill_offset_ms"] = (r["kill_wall"] - r["release_wall"]) * 1000
                    self.run.fire(KIND, self.fault["turn"], **{k: v for k, v in r.items() if k != "turn"})
                    if not self.wait_for_exit():
                        self.invalid("host group did not exit")
                        break
                    self.witness_id = rows[0][0]  # In memory only; never part of the evidence record.
                    self.witness_rows = rows
                    if self.ids and len({row[1] for row in rows}) != 1:
                        self.invalid("witness rows in several sessions")
                        break
                    self.validate(rows[0][1])
                    break
                time.sleep(0.002)
            else:
                self.invalid("poll timeout: no new host reply")
        except Exception as exc:
            self.invalid("poll/validation: " + type(exc).__name__)
        finally:
            try:
                self.stop_invalid_tool_loop()
            finally:
                self.finish()

    def wait_for_exit(self):
        end = self.record["kill_monotonic_ns"] / 1_000_000_000 + 5
        while time.monotonic() < end:
            exited = self.run.proc.process.poll() is not None
            try:
                os.killpg(self.run.proc.pid, 0)
            except ProcessLookupError:
                if exited:
                    self.record["exit_confirmed_ms"] = (
                        time.monotonic_ns() - self.record["kill_monotonic_ns"]) / 1_000_000
                    return True
            except PermissionError:
                pass  # A group we cannot signal is still present.
            time.sleep(0.005)
        return False

    def validate(self, session):
        r, current, seen = self.record, session, set()
        r["session"] = session[:12]
        while True:
            if current in seen:
                return self.invalid("lineage cycle")
            seen.add(current)
            row = self.host.execute("SELECT parent_session_id FROM sessions WHERE id = ?", (current,)).fetchone()
            if row is None:
                return self.invalid("lineage missing parent")
            if current == session:
                r["parent"] = row[0][:12] if row[0] else None
            if not row[0]:
                break
            current = row[0]
        r.update(root=current[:12], depth=len(seen) - 1)
        if self.ids:
            marks = ','.join('?' for _ in self.ids)
            rows = self.host.execute("SELECT id, session_id, tool_call_id FROM messages WHERE id > ? AND role = 'tool' "
                                     f"AND tool_call_id IN ({marks}) ORDER BY id", (self.max_id, *self.ids)).fetchall()
            calls = self.host.execute("SELECT tool_calls FROM messages WHERE id > ? AND session_id = ? "
                                      "AND role = 'assistant' AND tool_calls IS NOT NULL", (self.max_id, session))
            r["host_has_row"] = rows == self.witness_rows and {row[2] for row in rows} == set(self.ids) and any(
                set(self.ids) <= {c.get("id") for c in json.loads(row[0])} for row in calls)
        else:
            r["host_has_row"] = bool(self.host.execute("SELECT 1 FROM messages WHERE id = ? AND role = 'assistant' "
                                                      "AND content = ?", (self.witness_id, self.text)).fetchone())
        store = readonly(self.run.home / "lcm.db")
        try:
            if self.ids:
                r["store_has_row"] = bool(store.execute(
                    f"SELECT 1 FROM messages WHERE (role = 'tool' AND tool_call_id IN ({marks})) OR "
                    "(role = 'assistant' AND (" + " OR ".join("instr(tool_calls, ?) > 0" for _ in self.ids) + "))",
                    (*self.ids, *self.ids)).fetchone())
            else:
                r["store_has_row"] = bool(store.execute("SELECT 1 FROM messages WHERE content = ?", (self.text,)).fetchone())
                if any(self.fault["turn"] in plan["turns"] for plan in self.run.cell.get("tool_plan", [])):
                    prefix = f"call_T{self.fault['turn']:02d}_"
                    r["store_turn_tool_rows"] = store.execute(
                        "SELECT COUNT(*) FROM messages WHERE (role = 'tool' AND tool_call_id LIKE ?) OR "
                        "(role = 'assistant' AND instr(tool_calls, ?) > 0)", (prefix + '%', prefix)).fetchone()[0]
        finally:
            store.close()
        if current != self.run.sid:
            self.invalid("wrong root")
        elif r["depth"] != self.fault["depth"]:
            self.invalid("depth mismatch")
        elif not r["host_has_row"] or r["store_has_row"]:
            self.invalid("post-kill host/store check")
        elif not r["invalid_reason"]:
            r["valid"] = True

    def verdict(self, rec, restart_exit):
        self.record["restart_exit"] = restart_exit
        if restart_exit != "done":
            self.invalid("restart did not finish done")
        rec["reply_fault"] = {**self.record, "polls": list(self.record["polls"])}
        if not self.record["valid"]:
            invalid_verdict(rec, self.record["invalid_reason"])
