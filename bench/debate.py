"""Record multiagent debate on GSM8K (Du et al., ICML 2024) with every debater on EvolveMAS.

Uses the prompts, agents and scoring from examples/debate.py and writes docs/runs/debate.json.

    python bench/debate.py                 # gpt-4.1-mini debaters, gpt-4o-mini MemoryAgent
    python bench/debate.py --dry-run       # scripted backend, no API calls
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "bench")]

import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location("debate_example", ROOT / "examples" / "debate.py")
D = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(D)
from mixed_common import RetryLLM, check_and_save  # noqa: E402
from recorder import Recorder, TracedEvolveMAS  # noqa: E402

from evolvemas import OpenAICompatLLM, ScriptedLLM  # noqa: E402

AGENT_MODEL, SMALL_MODEL, MEMORY_MODEL = "gpt-4.1-mini", "gpt-4.1-nano", "gpt-4o-mini"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--out", default=str(ROOT / "docs/runs/debate.json"))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--data", default=None, help="GSM8K jsonl (default: examples/data/gsm8k_test_first10.jsonl)")
    ap.add_argument("--baseline", action="store_true", help="plug-in attached with online and evolve off")
    ap.add_argument("--long-out", default=None, help="also write a compact per-problem summary (results + knowledge)")
    args = ap.parse_args()

    problems = D.load_problems(args.data) if args.data else D.load_problems()
    if args.dry_run:
        agent_llm = small_llm = mem_llm = ScriptedLLM(D.mock_respond_factory(problems))
    else:
        import os
        key = os.environ["OPEN_AI_KEY"]
        agent_llm = RetryLLM(OpenAICompatLLM(AGENT_MODEL, api_key=key, temperature=0.2))
        small_llm = RetryLLM(OpenAICompatLLM(SMALL_MODEL, api_key=key, temperature=0.2))
        mem_llm = RetryLLM(OpenAICompatLLM(MEMORY_MODEL, api_key=key, temperature=0.2))

    rec = Recorder(
        "debate", "Multiagent debate · GSM8K",
        source={"dataset": "GSM8K test set (openai/gsm8k), problems 0-9 in order",
                "protocol": "Du et al., ICML 2024: 3 agents, 2 rounds (answer, then update after reading the others), majority vote",
                "description": "Three agents solve a grade-school math word problem, read each other's solutions, update their answers, and vote. The answer is checked against the GSM8K answer key.",
                "notes": "The team is heterogeneous by our choice for the demo: agent_c runs a smaller model and is told to answer quickly without writing out steps."},
        models={"agent_a and agent_b": "scripted (dry run)" if args.dry_run else AGENT_MODEL,
                "agent_c": "scripted (dry run)" if args.dry_run else SMALL_MODEL,
                "memory_agent": "scripted (dry run)" if args.dry_run else MEMORY_MODEL})
    for a, prof in D.AGENTS.items():
        rec.agent(a, label=a.replace("agent_", "Agent ").title(),
                  role={"agent_a": "careful solver", "agent_b": "careful solver", "agent_c": "answers fast, skips steps"}[a])
    store = Path(tempfile.mkdtemp(prefix="debate_mem_"))
    plugin = TracedEvolveMAS(store, mem_llm, rec, online=not args.baseline, evolve=not args.baseline)
    for a, prof in D.AGENTS.items():
        plugin.register(a, profile=prof)

    if args.episodes > len(problems):
        raise SystemExit(f"only {len(problems)} problems available")
    for ep in range(args.episodes):
        p = problems[ep % len(problems)]
        rec.episode({"problem": p["id"], "question": p["question"]})
        rec.env("Problem " + p["id"] + ":\n" + p["question"], visible_to=list(D.AGENTS))
        for a in D.AGENTS:
            plugin.begin_episode(a)
        replies, previous = [], {}
        for r in range(D.ROUNDS):
            current = {}
            for a in D.AGENTS:
                ctx = plugin.context(a)
                prompt = D.prompt_for(a, p["question"], r, previous, ctx)
                reply = (small_llm if a == "agent_c" else agent_llm)(D.SYSTEM, prompt).strip()
                rec.act(a, prompt, reply)
                current[a] = reply
            for a in D.AGENTS:      # each round's solutions are shared with everyone
                rec.msg(a, [b for b in D.AGENTS if b != a], f"Round {r + 1} (answer {D.extract(current[a])}):\n{current[a]}")
            for a in D.AGENTS:
                plugin.observe(a, [{"speaker": b, "content": f"round {r + 1}: {current[b]}"} for b in D.AGENTS])
            replies.append(current)
            previous = current
        res = D.evaluate(p, replies)
        res["summary"] = ("team correct" if res["team_correct"] else "team wrong") + f" · {res['first_round_correct']}/3 first answers right"
        rec.result(res)
        for a in D.AGENTS:
            plugin.end_episode(a, res)
        print(f"episode {ep}: {p['id']} gold={res['gold']} team={res['team_answer']} "
              f"first-round {res['first_round_correct']}/3 "
              + " ".join(f"{a}:{v['round1']}->{v['final']}" for a, v in res["answers"].items()), flush=True)

    usage = {"agents": dict(agent_llm.usage)} if args.dry_run else {"agent_a and agent_b": dict(agent_llm.usage), "agent_c": dict(small_llm.usage), "memory_agent": dict(mem_llm.usage)}
    calls = sum(u.get("calls", 0) for u in usage.values()) if not args.dry_run else agent_llm.usage["calls"]
    tin = sum(u.get("input_tokens", 0) for u in usage.values())
    tout = sum(u.get("output_tokens", 0) for u in usage.values())
    check_and_save(rec, args.out, {"calls": calls, "input_tokens": tin, "output_tokens": tout, "by_model": usage})
    if args.long_out:
        write_long(rec.data, args.long_out, baseline=args.baseline)
    print("usage:", usage)


def write_long(data, path, baseline):
    """Compact view of a long run: each problem's result and every agent's knowledge after it."""
    import json
    eps = []
    for ep in data["episodes"]:
        ends = {e["agent"]: e["state"] for e in ep["events"] if e["t"] == "hook" and e["name"] == "end_episode"}
        r = ep["result"]
        eps.append({
            "problem": r["problem"], "gold": r["gold"], "team_answer": r["team_answer"], "team_correct": r["team_correct"],
            "answers": {a: {k: v[k] for k in ("round1", "final", "round1_correct", "final_correct")} for a, v in r["answers"].items()},
            "knowledge": {a: {"strategies": s["strategies"], "partners": s["partners"],
                              "episodic": len(s["episodic"]), "semantic": len(s["semantic"])} for a, s in ends.items()},
        })
    out = {k: data[k] for k in ("benchmark", "title", "source", "models", "agents", "recorded_at", "usage")}
    out["condition"] = "EvolveMAS off (baseline)" if baseline else "EvolveMAS on"
    out["episodes"] = eps
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"saved {path}")


if __name__ == "__main__":
    main()
