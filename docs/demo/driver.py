"""Browser driver for the live demo. Runs inside Pyodide in a Web Worker.

It drives the multiagent debate host in examples/debate.py, using the real
`evolvemas` package, and streams one JSON event per step back to the page.
"""
from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path

sys.path[:0] = ["/py", "/py/examples"]

import debate as D  # noqa: E402
from evolvemas import EvolveMAS, ScriptedLLM  # noqa: E402

try:
    from js import postMessage  # Web Worker global
except ImportError:  # running under CPython for tests
    postMessage = None

_sink = None
RUNS = Path("/runs")


def emit(event):
    if _sink is not None:
        _sink(event)
    elif postMessage is not None:
        postMessage(json.dumps(event))


class HttpLLM:
    """Live backend over synchronous XHR. Synchronous requests are allowed in workers,
    which lets the plugin keep its plain `llm(system, user) -> str` interface."""

    def __init__(self, provider, api_key, model, base_url="", effort="low"):
        self.provider = provider
        self.api_key = api_key
        self.model = model
        self.base_url = (base_url or "").rstrip("/")
        self.effort = effort
        self.usage = {"input_tokens": 0, "output_tokens": 0, "calls": 0}

    def __call__(self, system, user):
        from js import XMLHttpRequest

        if self.provider == "anthropic":
            url = (self.base_url or "https://api.anthropic.com") + "/v1/messages"
            body = {"model": self.model, "max_tokens": 8000, "system": system,
                    "messages": [{"role": "user", "content": user}]}
            if not self.model.startswith("claude-haiku"):   # effort is not supported on Haiku 4.5
                body["output_config"] = {"effort": self.effort}
            headers = {"content-type": "application/json", "x-api-key": self.api_key,
                       "anthropic-version": "2023-06-01",
                       "anthropic-dangerous-direct-browser-access": "true"}
        else:
            url = (self.base_url or "https://api.openai.com/v1") + "/chat/completions"
            body = {"model": self.model, "temperature": 0.2,
                    "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
            headers = {"content-type": "application/json", "authorization": f"Bearer {self.api_key}"}

        xhr = XMLHttpRequest.new()
        xhr.open("POST", url, False)
        for k, v in headers.items():
            xhr.setRequestHeader(k, v)
        try:
            xhr.send(json.dumps(body))
        except Exception as e:  # network or CORS failure
            raise RuntimeError(f"Could not reach {url}. Check the base URL and your connection. ({e})")
        if xhr.status != 200:
            raise RuntimeError(f"{url} returned HTTP {xhr.status}: {str(xhr.responseText)[:300]}")
        data = json.loads(xhr.responseText)
        self.usage["calls"] += 1
        if self.provider == "anthropic":
            u = data.get("usage") or {}
            self.usage["input_tokens"] += u.get("input_tokens", 0)
            self.usage["output_tokens"] += u.get("output_tokens", 0)
            if data.get("stop_reason") == "refusal":
                raise RuntimeError(f"The model declined this request: {data.get('stop_details')}")
            return "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
        u = data.get("usage") or {}
        self.usage["input_tokens"] += u.get("prompt_tokens", 0)
        self.usage["output_tokens"] += u.get("completion_tokens", 0)
        return data["choices"][0]["message"]["content"] or ""


class Session:
    """Multiagent debate on GSM8K (examples/debate.py) with every debater on EvolveMAS."""

    def __init__(self, backend, evolve):
        self.problems = D.load_problems()
        if backend.get("kind") == "live":
            self.llm = HttpLLM(backend["provider"], backend["api_key"], backend["model"],
                               backend.get("base_url", ""))
        else:
            self.llm = ScriptedLLM(D.mock_respond_factory(self.problems))
        self.root = RUNS / uuid.uuid4().hex[:8]
        self.plugin = EvolveMAS(self.root, self.llm, online=evolve, evolve=evolve)
        for a, prof in D.AGENTS.items():
            self.plugin.register(a, profile=prof)
        self.episode = 0

    def files(self, agent):
        d = self.root / agent
        read = lambda name: (d / name).read_text(encoding="utf-8") if (d / name).exists() else ""
        return {"agent": agent, "memory_md": read("MEMORY.md"), "knowledge_md": read("KNOWLEDGE.md")}

    def run_episode(self):
        p = self.problems[self.episode % len(self.problems)]
        emit({"t": "episode", "ep": self.episode})
        emit({"t": "problem", "id": p["id"], "question": p["question"], "gold": p["answer"]})

        def on_event(e):
            if e["t"] == "context":
                emit({"t": "hook", "name": "context", "agent": e["agent"], "context": e["context"]})
            elif e["t"] == "say":
                emit({"t": "say", "who": e["agent"], "round": e["round"], "text": e["text"], "answer": e["answer"]})
            elif e["t"] in ("observe", "end_episode"):
                emit({"t": "hook", "name": e["t"], **self.files(e["agent"])})
            elif e["t"] == "result":
                emit({"t": "result", "ep": self.episode, "result": e["result"]})

        D.run_episode(self.plugin, self.llm, p, on_event)
        emit({"t": "usage", "usage": dict(self.llm.usage)})
        self.episode += 1


_session = None


def handle(msg_json):
    global _session
    msg = json.loads(msg_json)
    try:
        if msg["cmd"] == "new":
            _session = Session(msg["backend"], msg["evolve"])
            emit({"t": "session", "evolve": msg["evolve"], "agents": list(D.AGENTS),
                  "files": {a: _session.files(a) for a in D.AGENTS}})
        elif msg["cmd"] == "run":
            for _ in range(int(msg.get("episodes", 1))):
                _session.run_episode()
        emit({"t": "idle"})
    except Exception as e:
        emit({"t": "error", "msg": str(e)})
        emit({"t": "idle"})
