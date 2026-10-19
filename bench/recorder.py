"""Trace recorder for the benchmark demos.

Wraps the real `EvolveMAS` plug-in so every hook call, every MemoryAgent / Evolve model
call, every message between agents and every result lands in one JSON trace. The website
replays these traces; nothing in them is written by hand.

Trace schema (one file per benchmark):

{
  "benchmark": "sotopia", "title": ..., "source": {...}, "models": {...},
  "agents": [{"id": ..., "label": ..., "role": ...}],
  "recorded_at": ISO time, "usage": {...},
  "episodes": [
    {"index": 0, "setup": {...}, "result": {...},
     "events": [
       {"seq": n, "t": "msg",    "from": a, "to": [b, ...], "content": str, "channel": "public"|"private"|"system"},
       {"seq": n, "t": "act",    "agent": a, "prompt": str, "reply": str},           # main agent model call
       {"seq": n, "t": "env",    "content": str, "data": {...}},                    # environment step (tests run, night action, ...)
       {"seq": n, "t": "hook",   "name": "begin_episode"|"context"|"observe"|"end_episode", "agent": a,
        "input": ..., "output": ..., "state": {...},                                # agent state after the hook
        "calls": [{"role": "memory"|"evolve", "user": str, "response": str}]},
       {"seq": n, "t": "result", "data": {...}}
     ]}
  ]
}
"""
from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from evolvemas import EvolveMAS

MAX_TEXT = 8000


def _clip(s: Any) -> Any:
    if isinstance(s, str) and len(s) > MAX_TEXT:
        return s[:MAX_TEXT] + f"\n…[{len(s) - MAX_TEXT} more characters]"
    return s


class Recorder:
    def __init__(self, benchmark: str, title: str, source: Dict[str, Any], models: Dict[str, str]):
        self.data: Dict[str, Any] = {
            "benchmark": benchmark, "title": title, "source": source, "models": models,
            "agents": [], "recorded_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
            "episodes": [], "usage": {},
        }
        self.seq = 0
        self._calls: Optional[List[Dict[str, Any]]] = None

    # ---- structure
    def agent(self, id: str, label: str = "", role: str = "", **extra) -> None:
        self.data["agents"].append({"id": id, "label": label or id, "role": role, **extra})

    def episode(self, setup: Dict[str, Any]) -> None:
        self.data["episodes"].append({"index": len(self.data["episodes"]), "setup": setup, "events": [], "result": None})

    def _ev(self, ev: Dict[str, Any]) -> Dict[str, Any]:
        self.seq += 1
        ev = {"seq": self.seq, **ev}
        self.data["episodes"][-1]["events"].append(ev)
        return ev

    # ---- events the host reports
    def msg(self, sender: str, to: List[str], content: str, channel: str = "public") -> None:
        self._ev({"t": "msg", "from": sender, "to": list(to), "content": _clip(content), "channel": channel})

    def act(self, agent: str, prompt: str, reply: str) -> None:
        self._ev({"t": "act", "agent": agent, "prompt": _clip(prompt), "reply": _clip(reply)})

    def env(self, content: str, **data) -> None:
        self._ev({"t": "env", "content": _clip(content), "data": data})

    def result(self, data: Dict[str, Any]) -> None:
        self.data["episodes"][-1]["result"] = data
        self._ev({"t": "result", "data": data})

    # ---- plug-in internals
    def tap(self, role: str, user: str, response: str) -> None:
        if self._calls is not None:
            self._calls.append({"role": role, "user": _clip(user), "response": _clip(response)})

    def save(self, path, usage: Optional[Dict[str, Any]] = None) -> None:
        if usage:
            self.data["usage"] = usage
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.data, ensure_ascii=False, indent=1), encoding="utf-8")


class _Tap:
    """Model wrapper that reports each plug-in model call to the recorder."""

    def __init__(self, llm, rec: Recorder, role: str):
        self.llm, self.rec, self.role = llm, rec, role

    def __call__(self, system: str, user: str) -> str:
        out = self.llm(system, user)
        self.rec.tap(self.role, user, out)
        return out


def state_view(st) -> Dict[str, Any]:
    return {
        "episode": st.episode, "turn": st.turn,
        "episodic": [{"episode": e.episode, "turn": e.turn, "summary": e.summary, "source": e.source} for e in st.episodic],
        "semantic": {k: {"value": v.value, "confidence": v.confidence, "scope": v.scope} for k, v in st.semantic.items()},
        "working": st.working,
        "strategies": list(st.strategies),
        "partners": {k: {"tendency": p.tendency, "reliability": p.reliability} for k, p in st.partners.items()},
    }


class TracedEvolveMAS(EvolveMAS):
    """The real plug-in, with every hook reported to a Recorder."""

    def __init__(self, store_dir, llm, rec: Recorder, *, evolve_llm=None, **kw):
        self.rec = rec
        super().__init__(store_dir, _Tap(llm, rec, "memory"), evolve_llm=_Tap(evolve_llm or llm, rec, "evolve"), **kw)

    def _hook(self, name, agent_id, fn, inp=None):
        self.rec._calls = []
        try:
            out = fn()
        finally:
            calls, self.rec._calls = self.rec._calls, None
        self.rec._ev({"t": "hook", "name": name, "agent": agent_id, "input": inp,
                      "output": _clip(out) if isinstance(out, str) else None,
                      "state": state_view(self.state(agent_id)), "calls": calls})
        return out

    def begin_episode(self, agent_id=None):
        for aid in [agent_id] if agent_id else self.agents():
            self._hook("begin_episode", aid, lambda a=aid: EvolveMAS.begin_episode(self, a))

    def context(self, agent_id):
        return self._hook("context", agent_id, lambda: EvolveMAS.context(self, agent_id))

    def observe(self, agent_id, events):
        events = list(events)
        shown = [e if isinstance(e, dict) else {"speaker": "env", "content": e} for e in events]
        return self._hook("observe", agent_id, lambda: EvolveMAS.observe(self, agent_id, events),
                          inp=[{"speaker": e.get("speaker", "env"), "content": _clip(e.get("content", ""))} for e in shown])

    def end_episode(self, agent_id, result):
        return self._hook("end_episode", agent_id, lambda: EvolveMAS.end_episode(self, agent_id, result), inp=result)
