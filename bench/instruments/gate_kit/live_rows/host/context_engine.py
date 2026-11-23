"""Dependency-free ContextEngine contract shared by the pinned hosts."""
import copy
import json
from abc import ABC, abstractmethod

class ContextEngine(ABC):
    last_prompt_tokens = 0
    last_completion_tokens = 0
    last_total_tokens = 0
    threshold_tokens = 0
    context_length = 0
    compression_count = 0
    threshold_percent = 0.75
    protect_first_n = 3
    protect_last_n = 6
    emit_automatic_compaction_status = True

    def __init__(self, **kwargs):
        self.compression_count = 0
        self.last_prompt_tokens = 0

    @property
    @abstractmethod
    def name(self):
        raise NotImplementedError

    @abstractmethod
    def update_from_response(self, usage):
        raise NotImplementedError

    @abstractmethod
    def should_compress(self, prompt_tokens=None):
        raise NotImplementedError

    @abstractmethod
    def compress(self, messages, current_tokens=None, focus_topic=None, force=False, memory_context=""):
        raise NotImplementedError

    def should_compress_info(self, prompt_tokens=None):
        return self.should_compress(prompt_tokens), None

    def should_compress_preflight(self, messages):
        return False

    def get_automatic_compaction_status_message(self, *, phase, default_message, **context):
        return default_message if self.emit_automatic_compaction_status else None

    def on_session_start(self, session_id, **kwargs):
        pass

    def on_session_end(self, session_id, messages):
        pass

    def on_session_reset(self):
        self.compression_count = 0
        self.last_prompt_tokens = 0
        self.last_completion_tokens = 0
        self.last_total_tokens = 0

    def get_tool_schemas(self):
        return []

    def handle_tool_call(self, name, args, **kwargs):
        return json.dumps({"error": f"Unknown context engine tool: {name}"})

    def get_status(self):
        last_prompt = max(self.last_prompt_tokens, 0)
        return {
            "last_prompt_tokens": last_prompt,
            "threshold_tokens": self.threshold_tokens,
            "context_length": self.context_length,
            "usage_percent": min(100, last_prompt / self.context_length * 100) if self.context_length else 0,
            "compression_count": self.compression_count,
        }

    def clone_for_agent(self):
        return copy.deepcopy(self)

    def update_model(self, model, context_length, base_url="", api_key="", provider="", api_mode=""):
        # Host threshold resolution imports its compressor; keep this
        # unit-test fallback dependency-free. LCMEngine overrides it.
        self.context_length = context_length
