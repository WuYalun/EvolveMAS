"""BBH, cooperative setting: three decentralized agents answer together (AgentNet-style).

Per question: in round 1 each agent answers from the question alone; in round 2 each agent
sees the other two agents' round-1 messages and revises; the team answer is the majority of
round-2 answers (a tie goes to agent_a). Every agent runs with the EvolveMAS plug-in.

    python bench/bbh.py --dry-run --out /tmp/bbh.json      # scripted models, no spend
    python bench/bbh.py --episodes 3 --out docs/runs/bbh.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
import urllib.request
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]

from evolvemas import OpenAICompatLLM, ScriptedLLM, prompts  # noqa: E402
from recorder import Recorder, TracedEvolveMAS  # noqa: E402

SUBTASK = "logical_deduction_five_objects"
INDICES = [0, 1, 2]
AGENT_MODEL, MEMORY_MODEL = "gpt-4.1-mini", "gpt-4o-mini"
AGENTS = {
    "agent_a": "Careful solver: lists every constraint and places objects one at a time.",
    "agent_b": "Fast checker: proposes a full ordering quickly, then verifies it against each statement.",
    "agent_c": "Skeptic: looks for the constraint others are most likely to misread.",
}
FORMAT = ("Think step by step briefly, then finish with a final line of exactly the form "
          "'So the answer is (X).' where X is one option letter.")
SYSTEM = "You are one member of a team of independent agents solving a reasoning question together."


def load_items():
    cache = HERE / "data" / "cache" / f"bbh_{SUBTASK}.json"
    if not cache.exists():
        url = ("https://datasets-server.huggingface.co/rows?dataset=lukaemon/bbh"
               f"&config={SUBTASK}&split=test&offset=0&length=20")
        rows = json.loads(urllib.request.urlopen(url, timeout=60).read())["rows"]
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps([{"index": r["row_idx"], **r["row"]} for r in rows], indent=1))
    rows = json.loads(cache.read_text())
    return [rows[i] for i in INDICES]


def extract(text):
    m = re.findall(r"answer is\s*\(?([A-G])\)?", text, re.I)
    if m:
        return f"({m[-1].upper()})"
    m = re.findall(r"\(([A-G])\)", text)
    return f"({m[-1].upper()})" if m else None


def norm(s):
    return re.sub(r"[\s()]", "", (s or "")).lower()


# ------------------------------------------------------------------ dry-run backend
def scripted(system, user):
    if system == prompts.ONLINE_SYSTEM:
        ans = re.findall(r"answer is \(([A-G])\)", user)
        return json.dumps({"episodic": [f"Teammate answered ({ans[-1]})"] if ans else [],
                           "semantic": [{"key": "team_answers", "value": ",".join(ans), "confidence": "low", "scope": "episode"}],
                           "working": "Compare my answer with teammates."})
    if system == prompts.EVOLVE_SYSTEM:
        return json.dumps({"strategies": ["Anchor on the absolute positions (leftmost, rightmost, third) first."],
                           "partners": {"agent_b": {"tendency": "answers quickly", "reliability": "medium - one episode"}}})
    return "Placing the fixed positions first. So the answer is (A)."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--out", default="docs/runs/bbh.json")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--token-cap", type=int, default=1_500_000)
    args = ap.parse_args()

    if args.dry_run:
        agent_llm = mem_llm = ScriptedLLM(scripted)
    else:
        key = os.environ["OPEN_AI_KEY"]
        agent_llm = OpenAICompatLLM(AGENT_MODEL, api_key=key, temperature=0.2)
        mem_llm = OpenAICompatLLM(MEMORY_MODEL, api_key=key, temperature=0.2)
    llms = [agent_llm] if agent_llm is mem_llm else [agent_llm, mem_llm]

    items = load_items()[: args.episodes]
    rec = Recorder("bbh", "BBH · logical deduction (five objects)",
                   source={"dataset": "BIG-Bench Hard (lukaemon/bbh)", "subtask": SUBTASK, "indices": INDICES[: args.episodes],
                           "protocol": "3 agents, 2 rounds (propose, revise after seeing peers), majority vote"},
                   models={"agents": "scripted" if args.dry_run else AGENT_MODEL,
                           "memory_agent": "scripted" if args.dry_run else MEMORY_MODEL})
    plugin = TracedEvolveMAS(Path(tempfile.mkdtemp(prefix="bbh_mem_")), mem_llm, rec)
    for aid, prof in AGENTS.items():
        plugin.register(aid, profile=prof)
        rec.agent(aid, label=aid.replace("_", " ").title(), role=prof)

    def ask(aid, user):
        reply = agent_llm(SYSTEM, user)
        rec.act(aid, user, reply)
        return reply

    for ep, item in enumerate(items):
        rec.episode({"task_id": f"bbh_{SUBTASK}_{item['index']:04d}", "question": item["input"], "gold": item["target"]})
        plugin.begin_episode()
        rec.env(f"Question {item['index']} is broadcast to all agents.", question=item["input"])
        r1 = {}
        for aid, prof in AGENTS.items():
            mem = plugin.context(aid)
            user = (f"Your role: {prof}\n\nQuestion:\n{item['input']}\n\n{FORMAT}"
                    + (f"\n\n[Your memory]\n{mem}" if mem else ""))
            r1[aid] = ask(aid, user)
            rec.msg(aid, [a for a in AGENTS if a != aid], r1[aid])
        for aid in AGENTS:
            plugin.observe(aid, [{"speaker": "env", "content": f"Question: {item['input']}"}]
                           + [{"speaker": a, "content": r1[a]} for a in AGENTS])
        r2 = {}
        for aid, prof in AGENTS.items():
            mem = plugin.context(aid)
            peers = "\n\n".join(f"{a} said:\n{r1[a]}" for a in AGENTS if a != aid)
            user = (f"Your role: {prof}\n\nQuestion:\n{item['input']}\n\nYour round-1 answer:\n{r1[aid]}\n\n"
                    f"Your teammates' round-1 answers:\n{peers}\n\nRevise if needed. {FORMAT}"
                    + (f"\n\n[Your memory]\n{mem}" if mem else ""))
            r2[aid] = ask(aid, user)
            rec.msg(aid, [a for a in AGENTS if a != aid], r2[aid])
        answers = {a: extract(t) for a, t in r2.items()}
        counts = Counter(v for v in answers.values() if v)
        top = max(counts.values()) if counts else 0
        winners = [v for v, c in counts.items() if c == top]
        if len(winners) == 1:
            final = winners[0]
        else:                                   # tie or no parsable answer: agent_a breaks it
            final = answers["agent_a"] or (winners[0] if winners else None)
        correct = norm(final) == norm(item["target"])
        rec.env(f"Majority vote: {final} (votes {dict(counts)}). Gold {item['target']}: {'correct' if correct else 'incorrect'}.",
                votes=answers, final=final, gold=item["target"])
        for aid in AGENTS:
            plugin.observe(aid, [{"speaker": a, "content": r2[a]} for a in AGENTS if a != aid]
                           + [{"speaker": "env", "content": f"Team answer {final}; the correct answer was {item['target']}."}])
        result = {"team_answer": final, "gold": item["target"], "correct": correct, "votes": answers}
        rec.result(result)
        for aid in AGENTS:
            plugin.end_episode(aid, {"team_correct": correct, "team_answer": final, "gold": item["target"],
                                     "my_answer": answers[aid], "my_answer_correct": norm(answers[aid]) == norm(item["target"])})
        tokens = sum(l.usage["input_tokens"] + l.usage["output_tokens"] for l in llms)
        print(f"episode {ep}: {final} vs {item['target']} -> {'correct' if correct else 'incorrect'}; tokens so far {tokens}", flush=True)
        if tokens > args.token_cap:
            print("token cap reached, stopping", flush=True)
            break

    usage = {"calls": sum(l.usage["calls"] for l in llms),
             "input_tokens": sum(l.usage["input_tokens"] for l in llms),
             "output_tokens": sum(l.usage["output_tokens"] for l in llms),
             "by_model": {("agents" if l is agent_llm else "memory_agent"): dict(l.usage) for l in llms}}
    rec.save(args.out, usage)
    print("usage:", usage)


if __name__ == "__main__":
    main()
