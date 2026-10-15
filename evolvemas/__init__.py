"""EvolveMAS: memory-governed test-time self-evolution as a plug-in for any multi-agent system."""
from .core import EvolveMAS
from .llm import LLM, AnthropicLLM, OpenAICompatLLM, ScriptedLLM, parse_json
from .store import AgentState

__version__ = "0.1.0"

__all__ = ["EvolveMAS", "LLM", "AnthropicLLM", "OpenAICompatLLM", "ScriptedLLM", "parse_json", "AgentState"]
