# Preserved scanning-ON refusal
Prior PR head4be7e931d4e1a3450114fda4c57ad0ff39e9e761; executed mergeaf7994c5d631f0d95e80789069aa3481dd920b2a.
Run: https://github.com/electricsheephq/lcm-x/actions/runs/40089321305
PM native command051, actual interactive=False update, raised:
`hermes_cli.plugins_cmd.PluginScanBlocked: Security scan blocked plugin install: Blocked (dangerous verdict, 1 critical of 640 findings (echo_pipe_exec)). --force does not override a dangerous verdict.`
This failed scanning-ON receipt is preserved unchanged with all raw outputs. It was never relabelled PASS. R4 also reruns and records the scanning-ON refusal as EXPECTED_NEGATIVE before the S1-C.4 exception arm; see current051.stdout and cells.json.
