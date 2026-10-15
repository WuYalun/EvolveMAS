"""Model backends for the MemoryAgent and the Evolve step.

The plugin only needs one thing from a backend: `llm(system, user) -> str`.
Any callable with that signature works, so hosts can pass their own client.
"""
from __future__ import annotations

import json
import re
from typing import Any, Callable, Optional, Protocol


class LLM(Protocol):
    def __call__(self, system: str, user: str) -> str: ...


class AnthropicLLM:
    """Claude via the official SDK. Reads ANTHROPIC_API_KEY (or an `ant auth login` profile)."""

    def __init__(self, model: str = "claude-opus-5-5", effort: str = "low",
                 max_tokens: int = 16000, client: Any = None):
        import anthropic
        self.client = client or anthropic.Anthropic()
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens
        self.usage = {"input_tokens": 0, "output_tokens": 0, "calls": 0}

    def __call__(self, system: str, user: str) -> str:
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_config={"effort": self.effort},
        )
        self.usage["calls"] += 1
        self.usage["input_tokens"] += resp.usage.input_tokens
        self.usage["output_tokens"] += resp.usage.output_tokens
        if resp.stop_reason == "refusal":
            raise RuntimeError(f"model refused: {getattr(resp, 'stop_details', None)}")
        return "".join(b.text for b in resp.content if b.type == "text")


class OpenAICompatLLM:
    """Any OpenAI-compatible chat endpoint (the setup the original experiments used)."""

    def __init__(self, model: str, base_url: Optional[str] = None, api_key: Optional[str] = None,
                 temperature: float = 0.2):
        from openai import OpenAI
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.usage = {"input_tokens": 0, "output_tokens": 0, "calls": 0}

    def __call__(self, system: str, user: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        self.usage["calls"] += 1
        if resp.usage:
            self.usage["input_tokens"] += resp.usage.prompt_tokens
            self.usage["output_tokens"] += resp.usage.completion_tokens
        return resp.choices[0].message.content or ""


class ScriptedLLM:
    """Offline backend: `respond(system, user) -> str` decides every reply. No network, no cost.

    Used by the tests and by `examples/negotiation.py --backend mock` to exercise the full
    plugin loop deterministically.
    """

    def __init__(self, respond: Callable[[str, str], str]):
        self.respond = respond
        self.calls = []
        self.usage = {"input_tokens": 0, "output_tokens": 0, "calls": 0}

    def __call__(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        self.usage["calls"] += 1
        return self.respond(system, user)


def parse_json(text: str) -> Any:
    """Lenient JSON extraction: accepts fenced blocks or prose around one JSON object."""
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if m:
        text = m.group(1)
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError(f"no JSON object in model output: {text[:200]!r}")
    return json.loads(text[start:end + 1])


def as_llm(fn: Callable[[str, str], str]) -> LLM:
    return fn  # type: ignore[return-value]
