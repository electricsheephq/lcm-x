"""Pinned a42baa75 outcomes. Loss labels grant no parity waiver.

A fix PR lists (case, mode or '*', stage, key, issue) in INTENDED_DIFFERENCES.
Each entry must differ; the next PR moves BASELINE to that merge commit and
deletes its entries and the '#2543 proven' labels of the cells its fix closed.
A fix that removes a row main stores twice, reorders main's rows in a session,
restores an identical row main lost, or rewrites any stored column but store_id,
must change the preservation guard in the same PR. Under a tables exemption,
other tables keep main's per-path row-count floor unless listed with a reason
in SHRINKING_TABLES; equal-count rewrites of other tables are not caught.
The PR fixing #2736, #2737 or #2738 also deletes its CONTROL_LOSSES label.
LCMX_DIFF_SESSION_END_BASELINE overrides the pin locally;
CI keeps the pinned SHA.
"""
BASELINE = 'a42baa75e39a0085685b2b368e234fc631bac995'
INTENDED_DIFFERENCES = []
SHRINKING_TABLES = {'ingest_views': 'ingest_ledger.py _prune_views keeps only the last two views per kind.'}
CELLS = {f'{shape}-{route}': '#2543 proven'
         for shape, routes in (
             ('wrapped-base64', ('cold-pairwise', 'cold-counted')),
             ('data-uri-key', ('cold-pairwise', 'cold-counted')),
             ('changed-spillover', ('direct', 'busy')),
             ('changed-hermes-results', ('direct', 'busy')))
         for route in routes}
CONTROL_LOSSES = {'rollback': '#2736', 'interrupt': '#2737', 'bypass': '#2738'}
ROUTES = ('stored', 'ledger-off', 'before-ledger', 'S3', 'migrated', 'failed-ledger-write', 'composite-part')
FAULTS = ('busy-prefix', 'append', 'commit', 'repeated-retry', 'shutdown-deadline',
          'missing-payload', 'restart-between-commits', 'rollback', 'interrupt', 'bypass')
