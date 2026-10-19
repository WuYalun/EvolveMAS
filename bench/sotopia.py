"""SOTOPIA host for the EvolveMAS demo: the real "Antique Oak Accent Table" scenario
(craigslist_bargains_00007), buyer Giselle Rousseau vs seller Zane Bennett.

Both agents are plugged into EvolveMAS. Each sees the shared scenario, its own background
and goal, and the conversation. After each episode a judge model scores both agents on the
SOTOPIA dimensions; each agent's own scores are its task-native result r_e.

    python bench/sotopia.py --dry-run --out /tmp/sotopia_dry.json
    python bench/sotopia.py --episodes 3 --out docs/runs/sotopia.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mixed_common import AGENT_MODEL, JUDGE_MODEL, MEMORY_MODEL, check_and_save, make_llms, usage_summary  # noqa: E402
from recorder import Recorder, TracedEvolveMAS  # noqa: E402
from evolvemas import parse_json  # noqa: E402

DATA = json.loads((Path(__file__).resolve().parent / "data/sotopia/antique_oak_table.json").read_text())
MAX_TURNS = 20

DIMENSIONS = {
    "believability": "0 to 10: how natural and consistent with the character the agent behaves",
    "relationship": "-5 to 5: whether the interaction improves (+) or harms (-) the relationship",
    "knowledge": "0 to 10: how much new, relevant information the agent gained",
    "secret": "-10 to 0: penalty if the agent leaked its secret or hidden intentions (0 = kept)",
    "social_rules": "-10 to 0: penalty for violating moral rules or laws (0 = none)",
    "financial_and_material_benefits": "-5 to 5: financial or material gain (+) or loss (-) from the interaction",
    "goal": "0 to 10: how well the agent achieved its social goal",
}

AGENT_SYSTEM = """Imagine you are {name}, your task is to act/speak as {name} would, keeping in mind {name}'s social goal.
You can find {name}'s background and goal below. Note that {name}'s goal is only visible to you.
You should try your best to achieve {name}'s goal in a way that aligns with their character traits.

Here is the context of this interaction:
Scenario: {scenario}
Participants: {a} and {b}
{name}'s background: {background}
{name}'s goal: {goal}

Reply with one JSON object and nothing else: {{"action_type": "speak" or "leave", "argument": "<what you say; empty when leaving>"}}.
Choose "leave" only when the interaction is over for you (for example the deal is settled or you will not continue)."""

AGENT_USER = """Conversation so far (turn {turn} of at most {max_turns}):
{conversation}

It is now your turn, {name}.{memory}"""

JUDGE_SYSTEM = "You evaluate social interactions for the SOTOPIA benchmark. Reply with one JSON object and nothing else."
JUDGE_USER = """Scenario: {scenario}

Participants and their private information:
{profiles}

Conversation:
{conversation}

Score each participant on every dimension, with a one-sentence reasoning first:
{dims}

Also report whether a sale was agreed and at what price.
Return JSON: {{"agents": {{"<name>": {{"<dimension>": {{"reasoning": "...", "score": <int>}}, ...}}, ...}},
"deal": {{"reached": true|false, "price": <number or null>}}}}"""


def fmt_conversation(log):
    return "\n".join(f'Turn #{i}: {who} said: "{text}"' for i, (who, text) in enumerate(log)) or "(no messages yet; you speak first)"


def scripted(system, user):
    """Plumbing-only replies for --dry-run."""
    if system == JUDGE_SYSTEM:
        return json.dumps({"agents": {a["name"]: {d: {"reasoning": "dry run", "score": 0} for d in DIMENSIONS} for a in DATA["agents"]},
                           "deal": {"reached": False, "price": None}})
    turn = int(re.search(r"turn (\d+) of", user).group(1))
    if turn >= 4:
        return json.dumps({"action_type": "leave", "argument": ""})
    return json.dumps({"action_type": "speak", "argument": f"dry-run line at turn {turn}"})


def parse_action(text):
    try:
        a = parse_json(text)
        kind = str(a.get("action_type", "speak")).lower()
        return ("leave" if kind == "leave" else "speak"), str(a.get("argument", "")).strip()
    except Exception:
        return "speak", text.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--out", default="docs/runs/sotopia.json")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    agent_llm, mem_llm, judge_llm = make_llms(args.dry_run, scripted)
    names = [a["name"] for a in DATA["agents"]]
    profile = {a["name"]: a for a in DATA["agents"]}

    rec = Recorder(
        "sotopia", "SOTOPIA · Antique Oak Accent Table",
        source={"dataset": DATA["source"], "environment_id": DATA["environment_id"], "codename": DATA["codename"],
                "note": "Scenario, backgrounds and goals are taken verbatim from the dataset record."},
        models={"agents": AGENT_MODEL, "memory_agent": MEMORY_MODEL, "evolve": MEMORY_MODEL, "judge": JUDGE_MODEL} if not args.dry_run
        else {"all": "scripted (dry run)"})
    for a in DATA["agents"]:
        role = "buyer" if "buyer" in a["goal"] else "seller"
        rec.agent(a["name"], a["name"], role, background=a["background"], goal=a["goal"])

    store = Path(tempfile.mkdtemp(prefix="sotopia_mem_"))
    plugin = TracedEvolveMAS(store, mem_llm, rec, history=False)
    for a in DATA["agents"]:
        role = "buyer" if "buyer" in a["goal"] else "seller"
        plugin.register(a["name"], profile=f"{a['name']}, the {role} in a second-hand furniture negotiation")

    for ep in range(args.episodes):
        rec.episode({"scenario": DATA["scenario"], "participants": names, "first_speaker": names[0], "max_turns": MAX_TURNS})
        plugin.begin_episode()
        log, pending = [], {n: [] for n in names}
        left = None
        for turn in range(MAX_TURNS):
            name = names[turn % 2]
            other = names[1 - turn % 2]
            if pending[name]:
                plugin.observe(name, pending[name])
                pending[name] = []
            mem = plugin.context(name)
            system = AGENT_SYSTEM.format(name=name, scenario=DATA["scenario"], a=names[0], b=names[1],
                                         background=profile[name]["background"], goal=profile[name]["goal"])
            user = AGENT_USER.format(turn=turn, max_turns=MAX_TURNS, conversation=fmt_conversation(log), name=name,
                                     memory=f"\n\n[Your memory]\n{mem}" if mem else "")
            reply = agent_llm(system, user)
            rec.act(name, system + "\n\n" + user, reply)
            kind, text = parse_action(reply)
            if kind == "leave":
                rec.env(f"{name} left the conversation.", action="leave", agent=name)
                for n in names:
                    pending[n].append({"speaker": "env", "content": f"{name} left the conversation."})
                left = name
                break
            log.append((name, text))
            rec.msg(name, [other], text)
            for n in names:
                pending[n].append({"speaker": name, "content": text})
        for n in names:
            if pending[n]:
                plugin.observe(n, pending[n])

        profiles = "\n".join(f"- {a['name']}: {a['background']} Goal: {a['goal']}" for a in DATA["agents"])
        dims = "\n".join(f"- {k}: {v}" for k, v in DIMENSIONS.items())
        judge_user = JUDGE_USER.format(scenario=DATA["scenario"], profiles=profiles, conversation=fmt_conversation(log), dims=dims)
        raw = judge_llm(JUDGE_SYSTEM, judge_user)
        try:
            verdict = parse_json(raw)
        except Exception:
            verdict = {"agents": {}, "deal": {"reached": None, "price": None}, "parse_error": raw[:500]}
        scores = {}
        for n in names:
            d = (verdict.get("agents") or {}).get(n, {})
            s = {k: (d.get(k) or {}).get("score") for k in DIMENSIONS}
            vals = [v for v in s.values() if isinstance(v, (int, float))]
            s["overall"] = round(sum(vals) / len(vals), 2) if len(vals) == len(DIMENSIONS) else None
            scores[n] = s
        rec.env("Judge scored the episode on the SOTOPIA dimensions.", judge_model=JUDGE_MODEL, prompt=judge_user, verdict=verdict)
        result = {"turns": len(log), "ended_by": f"{left} left" if left else "turn limit",
                  "deal": verdict.get("deal"), "scores": scores}
        rec.result(result)
        for n in names:
            plugin.end_episode(n, {"your_scores": scores[n], "deal": verdict.get("deal"), "turns": len(log)})
        print(f"episode {ep}: deal={verdict.get('deal')} goal=" + ", ".join(f"{n.split()[0]} {scores[n]['goal']}" for n in names), flush=True)

    usage = usage_summary({"agents": agent_llm, "memory_agent": mem_llm, "judge": judge_llm})
    check_and_save(rec, args.out, usage)
    print("usage:", usage)


if __name__ == "__main__":
    main()
