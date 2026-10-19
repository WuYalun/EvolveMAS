"""EvolveMAS as a plug-in: three hooks around any host multi-agent system.

    ctx = plugin.context(agent_id)                 # before the agent acts: [M; K] to inject
    plugin.observe(agent_id, events)               # after a turn: evidence -> MA_online -> M
    plugin.end_episode(agent_id, result)           # after Evaluate: D_a(M, r; K) -> K

The host owns everything task-specific: which agents exist, who sees which message,
and how the episode is scored. The plugin only sees what the host hands to it.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

from . import prompts
from .llm import LLM, parse_json
from .store import AgentState, AgentStore, EpisodicEntry, PartnerProfile, SemanticFact

Event = Union[str, Dict[str, Any]]
log = logging.getLogger("evolvemas")


class EvolveMAS:
    def __init__(
        self,
        store_dir: Union[str, Path],
        llm: LLM,
        *,
        evolve_llm: Optional[LLM] = None,
        online: bool = True,          # Within-Episode Online Adaptation (MA_online each turn)
        evolve: bool = True,          # Cross-Episode Knowledge Evolution (D_a at episode end)
        max_episodic: int = 20,
        max_strategies: int = 8,
        inject_max_chars: int = 4000,
        history: bool = True,         # append a full snapshot to history.jsonl on every save
        fact_ttl: Optional[int] = 5,  # drop persistent facts not re-observed for this many episodes (None keeps them)
    ):
        self.root = Path(store_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.memory_llm = llm
        self.evolve_llm = evolve_llm or llm
        self.online = online
        self.evolve = evolve
        self.max_episodic = max_episodic
        self.max_strategies = max_strategies
        self.inject_max_chars = inject_max_chars
        self.history = history
        self.fact_ttl = fact_ttl
        self._stores: Dict[str, AgentStore] = {}
        self._states: Dict[str, AgentState] = {}

    # ------------------------------------------------------------------ setup
    def register(self, agent_id: str, profile: str = "") -> None:
        """Pair a main agent with its own MemoryAgent and isolated store. Resumes if the store exists."""
        store = AgentStore(self.root, agent_id, history=self.history)
        st = store.load() or AgentState(agent_id=agent_id, profile=profile)
        if profile:
            st.profile = profile
        self._stores[agent_id] = store
        self._states[agent_id] = st
        store.save(st, "register")

    def agents(self) -> List[str]:
        return list(self._states)

    def state(self, agent_id: str) -> AgentState:
        return self._states[agent_id]

    # ------------------------------------------------------------------ hooks
    def begin_episode(self, agent_id: Optional[str] = None) -> None:
        """Working Memory and episode-scoped Semantic facts are refreshed; Episodic, persistent Semantic and K persist.

        Dropping episode-scoped facts matters: without it, "deal accepted at $45" from the last
        episode is injected into the next one and the agent acts as if the deal already happened.
        """
        for aid in [agent_id] if agent_id else self.agents():
            st = self._states[aid]
            st.working = ""
            st.semantic = {k: v for k, v in st.semantic.items()
                           if v.scope != "episode" and (self.fact_ttl is None or st.episode - v.last_seen <= self.fact_ttl)}
            st.turn = 0
            self._stores[aid].save(st, "begin_episode")

    def context(self, agent_id: str) -> str:
        """Text block to put into the agent's prompt before it acts: [M_{e,t-1}; K_{e-1}].

        Sections are ordered from most to least specific to the current turn, so that
        truncation drops older, more general knowledge before the live episode state.
        """
        st = self._states[agent_id]
        parts: List[str] = []
        if st.working:
            parts.append("Working memory for this episode:\n" + st.working)
        ep_facts = {k: v for k, v in st.semantic.items() if v.scope == "episode"}
        facts = {k: v for k, v in st.semantic.items() if v.scope != "episode"}
        if ep_facts:
            parts.append("This episode so far:\n" + _fmt_facts(ep_facts))
        if facts:
            parts.append("Known facts:\n" + _fmt_facts(facts))
        if st.strategies:
            parts.append("Interaction Strategies (learned from earlier episodes):\n" + _bullets(st.strategies))
        if st.partners:
            parts.append("Partner Profiles:\n" + "\n".join(
                f"- {n}: {p.tendency} (reliability: {p.reliability})" for n, p in st.partners.items()))
        text = "\n\n".join(parts)
        if len(text) > self.inject_max_chars:
            text = text[: self.inject_max_chars].rsplit("\n", 1)[0] + "\n...[memory truncated]"
        return text

    def observe(self, agent_id: str, events: Iterable[Event]) -> None:
        """Append evidence the agent actually saw to its trajectory, then run MA_online."""
        st = self._states[agent_id]
        store = self._stores[agent_id]
        recs = [_normalize(e, st.episode, st.turn + 1) for e in events]
        if not recs:
            return
        st.turn += 1
        lines = store.append_trajectory(recs)
        if not self.online:
            store.save(st, "observe")
            return
        evidence = "\n".join(f"[L{ln}] {r['speaker']}: {r['content']}" for ln, r in zip(lines, recs))
        out = self._ask(self.memory_llm, prompts.ONLINE_SYSTEM, prompts.ONLINE_USER.format(
            agent_id=agent_id, profile=st.profile or "-",
            episodic=_fmt_episodic(st.episodic[-10:]), semantic=_fmt_semantic(st.semantic),
            working=st.working or "(empty)", episode=st.episode, turn=st.turn, evidence=evidence),
            what=f"MA_online for {agent_id}")
        if out is not None:
            speakers = ", ".join(dict.fromkeys(str(r["speaker"]) for r in recs))
            self._merge_online(st, out, source=speakers, ref=lines[-1])
        store.save(st, "observe")

    def end_episode(self, agent_id: str, result: Any) -> None:
        """Evaluate has produced r_e; distill K_e = D_a(M_e, r_e; K_{e-1}) and close the episode."""
        st = self._states[agent_id]
        store = self._stores[agent_id]
        st.results.append({"episode": st.episode, "result": result})
        if self.evolve:
            out = self._ask(self.evolve_llm, prompts.EVOLVE_SYSTEM, prompts.EVOLVE_USER.format(
                agent_id=agent_id, profile=st.profile or "-", episode=st.episode,
                result=json.dumps(result, ensure_ascii=False) if not isinstance(result, str) else result,
                episodic=_fmt_episodic([e for e in st.episodic if e.episode == st.episode]) or "(none)",
                semantic=_fmt_semantic(st.semantic),
                strategies=_bullets(st.strategies) or "(none)",
                partners="\n".join(f"- {n}: {p.tendency} / {p.reliability}" for n, p in st.partners.items()) or "(none)",
                max_strategies=self.max_strategies), what=f"D_a for {agent_id}")
            if out is not None:
                self._merge_evolve(st, out)
        st.episode += 1
        st.turn = 0
        store.save(st, "end_episode")

    # ------------------------------------------------------------ host helpers
    def wrap(self, agent_id: str, respond: Callable[[str], str]) -> Callable[[str], str]:
        """Wrap a host agent `respond(prompt) -> reply` so context injection happens automatically.

        The wrapped function prepends this agent's memory block to the prompt. Observation is
        still the host's call, because only the host knows what each agent is allowed to see.
        """
        def wrapped(prompt: str) -> str:
            ctx = self.context(agent_id)
            return respond(f"{prompt}\n\n[Your memory]\n{ctx}" if ctx else prompt)
        return wrapped

    # ---------------------------------------------------------------- internals
    @staticmethod
    def _ask(llm: LLM, system: str, user: str, what: str) -> Optional[Dict[str, Any]]:
        """Call the model and parse one JSON object. A bad reply skips this update instead of
        stopping the host's episode."""
        try:
            out = parse_json(llm(system, user))
        except (ValueError, json.JSONDecodeError) as e:
            log.warning("%s: unparseable reply, update skipped (%s)", what, e)
            return None
        if not isinstance(out, dict):
            log.warning("%s: expected a JSON object, got %s; update skipped", what, type(out).__name__)
            return None
        return out

    def _merge_online(self, st: AgentState, out: Dict[str, Any], source: str, ref: int) -> None:
        seen = {e.summary.strip().lower() for e in st.episodic}
        for s in _as_list(out.get("episodic")):
            s = _TAG.sub("", str(s or "")).strip()
            if s and s.lower() not in seen:
                st.episodic.append(EpisodicEntry(episode=st.episode, turn=st.turn, summary=s, source=source, ref=ref))
                seen.add(s.lower())
        st.episodic = st.episodic[-self.max_episodic:]
        for f in _as_list(out.get("semantic")):
            if not isinstance(f, dict):
                log.warning("MA_online: semantic item is not an object, skipped: %r", f)
                continue
            key = str(f.get("key") or "").strip()
            if not key:
                continue
            prev = st.semantic.get(key)
            conf = f.get("confidence")
            st.semantic[key] = SemanticFact(
                value=str(f.get("value", "")).strip(),
                confidence=conf if conf in ("low", "medium", "high") else "low",
                first_seen=prev.first_seen if prev else st.episode,
                last_seen=st.episode,
                scope="episode" if f.get("scope") == "episode" else "persistent",
            )
        if isinstance(out.get("working"), str) and out["working"].strip():
            st.working = out["working"].strip()

    def _merge_evolve(self, st: AgentState, out: Dict[str, Any]) -> None:
        # A key the model left out means "no revision", not "delete everything".
        if "strategies" in out and isinstance(out["strategies"], list):
            st.strategies = [str(s).strip() for s in out["strategies"] if s and str(s).strip()][: self.max_strategies]
        partners = out.get("partners")
        if isinstance(partners, dict):
            for name, p in partners.items():
                if isinstance(p, dict):
                    st.partners[str(name)] = PartnerProfile(tendency=str(p.get("tendency", "")),
                                                            reliability=str(p.get("reliability", "")))


_TAG = re.compile(r"^(\s*\[ep\d+\s+t\d+\]\s*)+")


def _as_list(x: Any) -> List[Any]:
    return x if isinstance(x, list) else []


def _normalize(e: Event, episode: int, turn: int) -> Dict[str, Any]:
    if isinstance(e, str):
        e = {"speaker": "env", "content": e}
    elif isinstance(e, (tuple, list)) and len(e) == 2:     # (speaker, content)
        e = {"speaker": str(e[0]), "content": str(e[1])}
    elif not isinstance(e, dict):
        e = {"speaker": "env", "content": str(e)}
    return {"episode": episode, "turn": turn, "speaker": e.get("speaker", "env"), "content": e.get("content", "")}


def _bullets(xs: List[str]) -> str:
    return "\n".join(f"- {x}" for x in xs)


def _fmt_facts(sem: Dict[str, SemanticFact]) -> str:
    return "\n".join(f"- {k}: {v.value} ({v.confidence})" for k, v in sem.items())


def _fmt_episodic(es: List[EpisodicEntry]) -> str:
    return "\n".join(f"- [ep{e.episode} t{e.turn}] {e.summary}" for e in es) or "(empty)"


def _fmt_semantic(sem: Dict[str, SemanticFact]) -> str:
    return "\n".join(f"- {k} = {v.value} ({v.confidence})" for k, v in sem.items()) or "(empty)"
