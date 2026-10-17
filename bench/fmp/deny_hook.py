"""PreToolUse hook for claude -p agentic runs: deny every tool that is not an lcmx MCP read tool.

An allowlist alone is not enough (claude -p auto-approves some read-only built-ins). Denials are
logged by tool name only, so the run counts them as tool violations.
"""
import json
import sys

event = json.load(sys.stdin)
name = str(event.get("tool_name") or "")
if not name.startswith("mcp__lcmx__"):
    with open(sys.argv[1], "a") as stream:
        stream.write(json.dumps({"tool": name}) + "\n")
    print(json.dumps({"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "deny",
                      "permissionDecisionReason": "Only the lcmx MCP tools are available in this session."}}))
