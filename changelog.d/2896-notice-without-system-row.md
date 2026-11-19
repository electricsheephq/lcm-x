- Fix: on a host whose list has no leading system row (the reference host's shape), the survival-fit notice now
  reaches the model. Rows a fit cuts there no longer leave with no pointer.
  - The notice is one marked block, `<lcm-survival-notice digest=…>`, right after the last verified summary part of
    the head: the re-formed carrier, or the summary row. It is never a new row. With a system row, the notice stays
    in the system row, byte-identical to before.
  - The summary verifier admits exactly that one block at that one position. Its digest binds it to the summary parts
    before it, their node ids and the notice text, so a hand-written, edited or copied block is ordinary text. Every
    reader of the head (carrier remainder, replay identity, de-dup, scaffold tests) therefore sees the same parts and
    remainder as without the block.
  - Two fits in a row leave one block. The block counts toward the fit's budget, and with its wrapper it stays within
    the 300-token cap; past the cap the block holds the counted notice instead.
  - The fit registers the summary row or re-formed carrier it writes, block included, as its emission. The carrier
    keys leave the block out. ignore_message_patterns never matches a carrier through its block.
  - With no system row and no summary head, the notice is not placed, and one warning is logged.
  - Known limits:
    - A user who pastes, byte for byte, a block the model echoed, right after the same standalone summary, has that
      pasted block read as LCM's.
    - A host that strips only the notice block from the list it sends back no longer matches the fit's emission. A
      cursor reset can then store the head row once more.
  - Raw counts (raw_tokens_outside_tail, source_token_count) include the block's tokens, at most 300, of the kind they
    already count; #2954 tracks them. (#2896)
