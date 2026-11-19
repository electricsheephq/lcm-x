- Fix: on a host whose list has no leading system row (the reference host's shape), the survival-fit notice now
  reaches the model when the head is a glued carrier. Rows a fit cuts there no longer leave with no pointer.
  - The fit first takes main's cut, on the list with any earlier block left out: the rows it keeps are main's on
    every input. The notice is then added on that same cut only, as one marked block, `<lcm-survival-notice
    digest=…>`, right after the last verified summary part of the cut's last glued carrier (the carrier the fit
    re-forms, or the last one the head already holds). It is never a new row, and a deeper cut is never taken to make
    room for it. With a system row, the notice stays in the system row, byte-identical to before.
  - A standalone summary row (no user row glued to it) gets no block, as before. The model reaches those rows on a
    later fit, once the host glues the summary into its next user row.
  - The summary verifier admits exactly that one block at that one position. Its digest binds it to the summary parts
    before it, their node ids and the notice text. A hand-written block, or one copied from another summary, is
    ordinary text. The digest is a public checksum, not a secret: someone who has the exact summary text and node ids
    can recompute it. Every reader of the head (carrier remainder, replay identity, de-dup, scaffold tests) therefore
    sees the same parts and remainder as without the block. After the block, only a `---` rule and a part header
    continue the summary; a header-shaped line after a blank line is the host row's own text.
  - Two fits in a row leave one block. The block counts toward the fit's budget, and with its wrapper it stays within
    the 300-token cap; past the cap the block holds the counted notice instead. The fit returns main's list when the
    notice does not fit beside the kept turns.
  - The fit registers the carrier it writes, block included, as its emission. The carrier keys leave the block out.
    ignore_message_patterns ignores a row only when its text matches both with and without its notice blocks (the
    fit's or pasted ones), so such a block can only keep a row.
  - When the notice is not placed (no glued carrier, or no room beside the kept turns), an applied fit logs one
    warning that names which; a fit that is planned but not applied logs none.
  - Known limits:
    - A user who pastes, byte for byte, a block the model echoed, right after the same summary, has that
      pasted block read as LCM's.
    - A host that strips only the notice block from the list it sends back no longer matches the fit's emission. A
      cursor reset can then store the head row once more.
  - Raw counts (raw_tokens_outside_tail, source_token_count) include the block's tokens, at most 300, of the kind they
    already count; #2954 tracks them. (#2896)
