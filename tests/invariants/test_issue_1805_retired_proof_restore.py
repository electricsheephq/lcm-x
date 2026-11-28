"""#1805 item 2: a store read that fails inside the #671 retry must not leave the durable proof retired."""

import sqlite3

from hermes_lcm.engine import LCMEngine
from tests.test_issue_1116_stub_first_exit_commit_proof import host_for as host_for, leave_proof_stale
from tests.test_issue_1116_stub_first_exit_commit_proof import stub_first_session as stub_first_session


def test_failed_read_in_the_retry_restores_the_retired_proof(stub_first_session, caplog, monkeypatch):
    host = stub_first_session("summary", stub_first=False)
    leave_proof_stale(host, caplog)
    find = LCMEngine._find_reconciled_cursor_for_store_tail
    failures = []

    def failing_find(self, *args, **kwargs):
        # The seam: only the retry (proof retired) reads a failing store.
        if getattr(self, "_retired_durable_proof", None) is not None:
            failures.append(1)
            raise sqlite3.OperationalError("disk I/O error")
        return find(self, *args, **kwargs)

    monkeypatch.setattr(LCMEngine, "_find_reconciled_cursor_for_store_tail", failing_find)
    host.cold_rebind()
    assert failures, "the retry never ran: the seam did not fire"
    assert getattr(host.engine, "_retired_durable_proof", None) is None
