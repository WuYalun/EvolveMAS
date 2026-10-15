"""Per-agent isolated storage: Structured Memory M = (E, S, W), Interaction Knowledge K = (G, P).

Layout under <root>/<agent_id>/:
    state.json         machine-readable M and K (the source of truth)
    MEMORY.md          human-readable render of M, rewritten on every save
    KNOWLEDGE.md       human-readable render of K
    trajectory.jsonl   raw evidence the agent observed (tau), append-only
    history.jsonl      one snapshot per update, for replay and audit (EvolveMAS(history=False) turns it off)
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class EpisodicEntry:
    episode: int
    turn: int
    summary: str
    source: str = ""          # who produced the evidence
    ref: int = 0              # trajectory.jsonl line number


@dataclass
class SemanticFact:
    value: str
    confidence: str = "low"   # low / medium / high
    first_seen: int = 0       # episode index
    last_seen: int = 0
    scope: str = "persistent" # "episode" facts are dropped when the next episode begins


@dataclass
class PartnerProfile:
    tendency: str
    reliability: str


@dataclass
class AgentState:
    agent_id: str
    profile: str = ""
    episode: int = 0          # index of the current (or next) episode
    turn: int = 0
    episodic: List[EpisodicEntry] = field(default_factory=list)
    semantic: Dict[str, SemanticFact] = field(default_factory=dict)
    working: str = ""
    strategies: List[str] = field(default_factory=list)
    partners: Dict[str, PartnerProfile] = field(default_factory=dict)
    results: List[Dict[str, Any]] = field(default_factory=list)

    def to_json(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, d: Dict[str, Any]) -> "AgentState":
        return cls(
            agent_id=d["agent_id"],
            profile=d.get("profile", ""),
            episode=d.get("episode", 0),
            turn=d.get("turn", 0),
            episodic=[EpisodicEntry(**e) for e in d.get("episodic", [])],
            semantic={k: SemanticFact(**v) for k, v in d.get("semantic", {}).items()},
            working=d.get("working", ""),
            strategies=list(d.get("strategies", [])),
            partners={k: PartnerProfile(**v) for k, v in d.get("partners", {}).items()},
            results=list(d.get("results", [])),
        )


_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 _.@-]*$")


def check_agent_id(agent_id: str) -> str:
    """Agent ids become directory names, so they must not reach outside the store root."""
    if not isinstance(agent_id, str) or not _SAFE_ID.match(agent_id) or ".." in agent_id:
        raise ValueError(f"invalid agent_id {agent_id!r}: use letters, digits, space, '_', '.', '@' or '-', "
                         "starting with a letter or digit, without '..'")
    return agent_id


class AgentStore:
    def __init__(self, root: Path, agent_id: str, history: bool = True):
        self.dir = Path(root) / check_agent_id(agent_id)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.agent_id = agent_id
        self.history = history
        self._state_path = self.dir / "state.json"
        self._traj_path = self.dir / "trajectory.jsonl"
        self._hist_path = self.dir / "history.jsonl"
        self._traj_len: Optional[int] = None

    # --- state -------------------------------------------------------------
    def load(self) -> Optional[AgentState]:
        if not self._state_path.exists():
            return None
        return AgentState.from_json(json.loads(self._state_path.read_text(encoding="utf-8")))

    def save(self, st: AgentState, event: str) -> None:
        self._state_path.write_text(json.dumps(st.to_json(), ensure_ascii=False, indent=1), encoding="utf-8")
        (self.dir / "MEMORY.md").write_text(render_memory(st), encoding="utf-8")
        (self.dir / "KNOWLEDGE.md").write_text(render_knowledge(st), encoding="utf-8")
        if self.history:
            with self._hist_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"event": event, "ts": time.time(), "state": st.to_json()}, ensure_ascii=False) + "\n")

    # --- trajectory --------------------------------------------------------
    def append_trajectory(self, records: List[Dict[str, Any]]) -> List[int]:
        start = self.trajectory_len()
        with self._traj_path.open("a", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        self._traj_len = start + len(records)
        return list(range(start + 1, start + 1 + len(records)))

    def trajectory_len(self) -> int:
        """Line count of trajectory.jsonl; read from disk once, then tracked in memory."""
        if self._traj_len is None:
            if not self._traj_path.exists():
                self._traj_len = 0
            else:
                with self._traj_path.open(encoding="utf-8") as f:
                    self._traj_len = sum(1 for _ in f)
        return self._traj_len

    def read_trajectory(self, episode: int) -> List[Dict[str, Any]]:
        if not self._traj_path.exists():
            return []
        out = []
        with self._traj_path.open(encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if r.get("episode") == episode:
                    out.append(r)
        return out


def render_memory(st: AgentState) -> str:
    lines = [f"# MEMORY · {st.agent_id}", "", f"episode: {st.episode} · turn: {st.turn}", "", "## Episodic"]
    for e in st.episodic:
        lines.append(f"- [ep{e.episode} t{e.turn}] {e.summary} (from {e.source or '?'}) [[trajectory.jsonl|L{e.ref}]]")
    lines += ["", "## Semantic"]
    for k, v in st.semantic.items():
        lines.append(f"- {k} = {v.value} ({v.confidence})")
    lines += ["", "## Working", st.working or "_empty_", ""]
    return "\n".join(lines)


def render_knowledge(st: AgentState) -> str:
    lines = [f"# KNOWLEDGE · {st.agent_id}", "", "## Interaction Strategies"]
    lines += [f"- {s}" for s in st.strategies] or ["_none yet_"]
    lines += ["", "## Partner Profiles"]
    for name, p in st.partners.items():
        lines += [f"### {name}", f"- tendency: {p.tendency}", f"- reliability: {p.reliability}"]
    if not st.partners:
        lines.append("_none yet_")
    return "\n".join(lines) + "\n"
