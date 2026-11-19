- Fix: the survival-fit notice now names every range of this session's stored rows that is still out of view after
  the fit, not only the range this fit dropped, so a second fit in a row no longer leaves the first fit's rows with no
  pointer. A row counts as in view when it, or a byte-equal stored copy of it, is in the returned list or under a
  summary that list shows. Each range comes with the exact call that loads it,
  `lcm_load_session(session_id="<id>", after_store_id=<first - 1>, limit=200)`, paging with next_cursor. The newest
  8 ranges are named, and any older ones are counted in one clause that points to lcm_grep. The cut rule is unchanged,
  but the longer notice counts toward the fit's budget, so at the budget's edge a fit can drop one more turn. The builder
  is `survival_fit_notice` in survival_fit.py. (#2877)
