"""Pinned a42baa75 outcomes. Labels describe loss; they grant no parity waiver."""
BASELINE = 'a42baa75e39a0085685b2b368e234fc631bac995'
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
