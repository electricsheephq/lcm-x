- Preserve eligible earlier queued user texts on whole-replace Hermes hosts as
  recall-only records, with exact parent/ordinal locators and counted overlap or
  scan-limit withholding. Refused captures and unfinished drains retain the known
  missing-text residual; canonical transcript rows stay unchanged.
- `lcm.db` retains its existing older-reader compatibility. The separate
  `lcm.db.records` file remains format 1 and requires a v0.30.2 records reader.
  Back up and restore the pair together; older canonical readers ignore records.
- Coordinate first records creation with session purges, refuse unmapped records
  tables during purge, and preserve backup pair identity across failed or
  interrupted publication, including an old backup with no records file.
