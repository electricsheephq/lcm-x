### Fixed
- Reliability B1 binds each `continue` reply by its host occurrence, including its timestamp, so lossless out-of-order head reconciliation is accepted while missing and surplus replies still fail.
