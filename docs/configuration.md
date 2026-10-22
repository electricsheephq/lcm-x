# Configuration

See [Configuration and activation](../skills/hermes-lcm/references/configuration.md)
for the complete configuration reference.

`LCM_SENDER_CAPTURE` is a non-field, profile-aware boolean toggle, enabled by
default. Set `LCM_SENDER_CAPTURE=0` to disable creation of pending sender
associations. The nullable sender columns are still added, and no historical
rows are backfilled. Capture records the gateway session principal as an
unverified label; it does not change stored or delivered text.
