"""Prompts for MA_online (within-episode) and D_a (cross-episode).

Both return JSON so the merge into memory is done by code, deterministically.
The rules carried over from the experiment code (improve/memory_agent/eee_hook.py):
partner profiles describe behaviour, never per-episode assignments; Semantic keeps
candidate facts at low confidence instead of dropping them.
"""

ONLINE_SYSTEM = """You are the MemoryAgent paired with one main agent in a multi-agent system.
You maintain that agent's Structured Memory from the interaction evidence it observed.
You never act in the task yourself. Reply with one JSON object and nothing else."""

ONLINE_USER = """Main agent: {agent_id}
Agent profile: {profile}

Current Structured Memory
-- Episodic (most recent last) --
{episodic}
-- Semantic --
{semantic}
-- Working --
{working}

New evidence observed at episode {episode}, turn {turn}:
{evidence}

Update the memory. Return JSON with exactly these keys:
{{
  "episodic": ["<one-line event summary>", ...],   // 0-2 NEW events from the evidence only; no repeats of existing entries
  "semantic": [{{"key": "<snake_case>", "value": "<fact>", "confidence": "low|medium|high", "scope": "episode|persistent"}}, ...],
                                                     // facts or regularities; reuse an existing key to revise it.
                                                     // scope=episode: true only in this episode (current offers, deal status, who said what).
                                                     // scope=persistent: should still hold next episode (item facts, partner regularities).
  "working": "<2-4 lines: current focus, latest position of each party, next step>"
}}
Rules: record what happened, not advice. A fact seen once is low confidence. Keep each line under 160 characters."""

EVOLVE_SYSTEM = """You distill reusable Interaction Knowledge for one agent after an episode ends.
Interaction Strategies are short, actionable rules that should transfer to future episodes.
Partner Profiles model how each peer tends to behave. Reply with one JSON object and nothing else."""

EVOLVE_USER = """Agent: {agent_id}
Agent profile: {profile}

Episode {episode} has ended. Task-native result r_e:
{result}

Structured Memory at episode end
-- Episodic --
{episodic}
-- Semantic --
{semantic}

Current Interaction Knowledge (from earlier episodes)
-- Strategies --
{strategies}
-- Partner Profiles --
{partners}

Revise the knowledge, conditioned on the result. Return JSON:
{{
  "strategies": ["<rule>", ...],        // the full revised list, at most {max_strategies}; keep rules that still hold,
                                         // rewrite or drop ones the result contradicts, add new ones the evidence supports
  "partners": {{"<peer name>": {{"tendency": "<behaviour across episodes>", "reliability": "high|medium|low - <reason>"}}}}
}}
Rules: a strategy must name a concrete action and when to use it, not a topic word.
Describe partners by behaviour only, never by roles or labels assigned for a single episode.
Only include peers that appear in the evidence."""
