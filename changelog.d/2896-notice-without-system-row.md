- Fix: on a host whose list has no leading system row (the reference host's shape), the survival-fit notice now
  reaches the model when the head is a glued carrier. Rows a fit cuts there no longer leave with no pointer.
  - The notice is one marked block, `<lcm-survival-notice digest=…>`, right after the last verified summary part of
    a verified glued carrier in the head: the carrier the fit re-forms, or the last one the head already holds. It is
    never a new row. With a system row, the notice stays in the system row, byte-identical to before.
  - A standalone summary row (no user row glued to it) gets no block, as before. The model reaches those rows on a
    later fit, once the host glues the summary into its next user row.
  - The summary verifier admits exactly that one block at that one position. Its digest binds it to the summary parts
    before it, their node ids and the notice text. A hand-written block, or one copied from another summary, is
    ordinary text. The digest is a public checksum, not a secret: someone who has the exact summary text and node ids
    can recompute it. Every reader of the head (carrier remainder, replay identity, de-dup, scaffold tests) therefore
    sees the same parts and remainder as without the block. After the block, only a `---` rule and a part header
    continue the summary; a header-shaped line after a blank line is the host row's own text.
  - Two fits in a row leave one block. The block counts toward the fit's budget, and with its wrapper it stays within
    the 300-token cap; past the cap the block holds the counted notice instead. When the block alone would push the
    whole-turn cut over budget, the fit returns the list it returned before (the summary kept, the notice not
    placed) rather than drop the summary.
  - The fit registers the carrier it writes, block included, as its emission. The carrier keys leave the block out.
    ignore_message_patterns never matches a row through a notice block (the fit's or a pasted one), so such a block
    can only keep a row.
  - With no system row and no glued carrier for the block, the notice is not placed, and an applied fit logs one
    warning (a fit that is planned but not applied logs none).
  - Known limits:
    - A user who pastes, byte for byte, a block the model echoed, right after the same summary, has that
      pasted block read as LCM's.
    - A host that strips only the notice block from the list it sends back no longer matches the fit's emission. A
      cursor reset can then store the head row once more.
  - Raw counts (raw_tokens_outside_tail, source_token_count) include the block's tokens, at most 300, of the kind they
    already count; #2954 tracks them. (#2896)
