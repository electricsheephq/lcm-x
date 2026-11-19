- Fix: the survival-fit notice now names every range of stored rows still out of view after the fit, not only the range
  this fit dropped, so a second fit in a row no longer leaves the first fit's rows with no pointer.
  - A row with a visible byte-equal copy (role, content, tool calls and call id) in the returned list, or under a
    summary that list shows, is not named, except rows this fit dropped.
  - The rows a fit drops are always named. A rotation's carried parent rows (its half-open carry ranges, read with one
    indexed query per range) are named under the parent's session id on every fit while they are out of view.
  - Each range gives `session_id`, `after_store_id`, its own `limit` (the range's width, at most 200) and where to
    stop, for `lcm_load_session` with `next_cursor`. A row marked `content_truncated` is read whole with
    `lcm_expand(store_id)`, following `next_content_offset` while `has_more` is set.
  - The newest 8 ranges are named, and older ones are counted in one clause that points to lcm_grep.
  - The notice holds no character JSON escapes. It stays within 300 tokens by the plugin's own counter: the oldest named
    ranges fold into the counted clause, and a single range whose session id alone passes the cap gets the counted
    notice with no call.
  - Trade-off: every named range carries its explicit session id, so a longer id makes the notice longer than the
    previous single-range notice and moves the fit's budget edge by about a quarter token per id character. Inside that
    window the newest user row is shown mark-only, the same cliff the previous notice has at its own edge. This affects
    the view only: no stored row is lost.
  - The cut rule is unchanged, but the longer notice counts toward the fit's budget. At the budget's edge a fit can
    drop one more turn, or take the emergency path that drops the summary prefix while keeping those rows in the list.
  - The builder is `survival_fit_notice` in survival_fit.py.
  - On a host whose list has no leading system row the notice is not placed, as before (#2896). (#2877)
