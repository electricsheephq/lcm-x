"""Ordered tool metadata shared by the two Hermes exposure paths.

Handler names are resolved on the tools module at dispatch time.
"""

from typing import NamedTuple

from . import schemas


class ToolSpec(NamedTuple):
    name: str
    schema: dict
    handler_name: str
    emoji: str
    dormant_unless: str | None


TOOL_REGISTRY = (
    ToolSpec("lcm_grep", schemas.LCM_GREP, "lcm_grep", "🔍", None),
    ToolSpec("lcm_recall", schemas.LCM_RECALL, "lcm_recall", "🧠", None),
    ToolSpec("lcm_query_state", schemas.LCM_QUERY_STATE, "lcm_query_state", "🧾", "assertions_enabled"),
    ToolSpec("lcm_compute", schemas.LCM_COMPUTE, "lcm_compute", "🧮", None),
    ToolSpec("lcm_compile_evidence", schemas.LCM_COMPILE_EVIDENCE, "lcm_compile_evidence", "🧷", None),
    ToolSpec("lcm_evidence_pack", schemas.LCM_EVIDENCE_PACK, "lcm_evidence_pack", "📦", None),
    ToolSpec("lcm_retrieve", schemas.LCM_RETRIEVE, "lcm_retrieve", "🧭", "adaptive_retrieval_enabled"),
    ToolSpec("lcm_recent", schemas.LCM_RECENT, "lcm_recent", "🕒", None),
    ToolSpec("lcm_load_session", schemas.LCM_LOAD_SESSION, "lcm_load_session", "📋", None),
    ToolSpec("lcm_describe", schemas.LCM_DESCRIBE, "lcm_describe", "📊", None),
    ToolSpec("lcm_expand", schemas.LCM_EXPAND, "lcm_expand", "🔎", None),
    ToolSpec("lcm_expand_query", schemas.LCM_EXPAND_QUERY, "lcm_expand_query", "❓", None),
    ToolSpec("lcm_status", schemas.LCM_STATUS, "lcm_status", "💚", None),
    ToolSpec("lcm_inspect", schemas.LCM_INSPECT, "lcm_inspect", "🧭", None),
    ToolSpec("lcm_doctor", schemas.LCM_DOCTOR, "lcm_doctor", "🏥", None),
)
