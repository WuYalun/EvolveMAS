"""Multiagent debate on GSM8K (Du et al., "Improving Factuality and Reasoning in Language
Models through Multiagent Debate", ICML 2024), with every debater attached to EvolveMAS.

Three agents answer a GSM8K problem independently, then each reads the others' solutions
and gives an updated answer. The team answer is the majority of final answers and is
checked against the GSM8K answer key. The prompts follow the paper's GSM8K prompts.

One debater (agent_c) is told to answer quickly without writing out steps, so the team is
not uniform: its partners can learn how reliable its first answers are, and it can learn
to change its own habit. That choice is ours, for the demo; the paper's debaters are copies
of one model.

    pip install -e ".[openai]"
    python examples/debate.py --backend openai --episodes 5        # real models
    python examples/debate.py --backend mock --episodes 5          # offline, no API key
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from pathlib import Path

from evolvemas import AnthropicLLM, EvolveMAS, OpenAICompatLLM, ScriptedLLM, prompts

DATA = Path(__file__).with_name("data") / "gsm8k_test_first10.jsonl"
ROUNDS = 2
AGENTS = {
    "agent_a": "You are a careful math solver.",
    "agent_b": "You are a careful math solver.",
    "agent_c": "You answer quickly and briefly. You usually skip writing out intermediate steps.",
}
SYSTEM = "You are one of several agents solving a math problem together."

FIRST = ("Can you solve the following math problem? {question} Explain your reasoning. "
         "Your final answer should be a single numerical number, in the form \\boxed{{answer}}, "
         "at the end of your response.")
DEBATE = ("These are the solutions to the problem from other agents: {others}\n\n"
          "Using the reasoning from other agents as additional advice, can you give an updated answer? "
          "Examine your solution and that other agents step by step. Your final answer should be a "
          "single numerical number, in the form \\boxed{{answer}}, at the end of your response.")


def load_problems(path=DATA):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def extract(text):
    """Last \\boxed{...} number, else the last number in the text."""
    m = re.findall(r"\\boxed\{([^}]*)\}", text or "")
    s = m[-1] if m else (re.findall(r"-?\d[\d,]*\.?\d*", text or "") or [""])[-1]
    s = s.replace(",", "").replace("$", "").strip()
    try:
        v = float(s)
        return str(int(v)) if v == int(v) else str(v)
    except ValueError:
        return s or None


def prompt_for(agent, question, round_idx, previous, memory=""):
    """previous: {agent_id: last reply} from the round before (round_idx >= 1)."""
    if round_idx == 0:
        user = FIRST.format(question=question)
    else:
        others = "".join(f"\n\n One agent solution: ```{previous[a]}```" for a in previous if a != agent)
        user = FIRST.format(question=question) + "\n\nYour previous answer: ```" + previous[agent] + "```\n\n" + DEBATE.format(others=others)
    user = AGENTS[agent] + "\n\n" + user
    if memory:
        user += "\n\n[Your memory]\n" + memory
    return user


def evaluate(problem, replies):
    """replies: list of rounds, each {agent: text}."""
    gold = problem["answer"]
    answers = {a: {"round1": extract(replies[0][a]), "final": extract(replies[-1][a])} for a in AGENTS}
    for a, v in answers.items():
        v["round1_correct"] = v["round1"] == gold
        v["final_correct"] = v["final"] == gold
    votes = Counter(v["final"] for v in answers.values())
    team = votes.most_common(1)[0][0]
    return {"problem": problem["id"], "gold": gold, "team_answer": team, "team_correct": team == gold,
            "first_round_correct": sum(v["round1_correct"] for v in answers.values()), "answers": answers}


def result_for(agent, result):
    """What end_episode() receives for one agent: its own outcome and its partners', without the
    answer key, so the knowledge it distills is about things it can act on next time."""
    me = result["answers"][agent]
    return {"problem": result["problem"],
            "my_first_answer_correct": me["round1_correct"], "my_final_answer_correct": me["final_correct"],
            "team_answer_correct": result["team_correct"],
            "partners": {b: {"first_answer_correct": v["round1_correct"], "final_answer_correct": v["final_correct"]}
                         for b, v in result["answers"].items() if b != agent}}


def run_episode(plugin, llm, problem, on_event=lambda e: None):
    """One debate. Every agent sees the question and, from round 2 on, the others' replies."""
    for a in AGENTS:
        plugin.begin_episode(a)
    replies, previous = [], {}
    for r in range(ROUNDS):
        current = {}
        for a in AGENTS:
            ctx = plugin.context(a)
            on_event({"t": "context", "agent": a, "context": ctx})
            reply = llm(SYSTEM, prompt_for(a, problem["question"], r, previous, ctx)).strip()
            current[a] = reply
            on_event({"t": "say", "agent": a, "round": r + 1, "text": reply, "answer": extract(reply)})
        for a in AGENTS:   # everyone sees this round's solutions before the next round
            plugin.observe(a, [{"speaker": b, "content": f"round {r + 1}: {current[b]}"} for b in AGENTS])
            on_event({"t": "observe", "agent": a})
        replies.append(current)
        previous = current
    result = evaluate(problem, replies)
    on_event({"t": "result", "result": result})
    for a in AGENTS:
        plugin.end_episode(a, result_for(a, result))
        on_event({"t": "end_episode", "agent": a})
    return result


# ------------------------------------------------------------------ mock backend
def mock_respond_factory(problems):
    """Scripted debaters and MemoryAgent. agent_a and agent_b are always right. agent_c's first
    answer is off unless its own injected knowledge tells it to write out each step; in round 2
    it adopts the majority. The plug-in itself is real: agent_c only changes because of what
    the plug-in wrote to its knowledge."""
    gold = {p["question"]: p["answer"] for p in problems}

    def find_gold(text):
        for q, g in gold.items():
            if q[:60] in text:
                return g
        return "0"

    def respond(system, user):
        if system == prompts.ONLINE_SYSTEM:
            got = re.findall(r"(agent_[abc]): round (\d): .*?\\boxed\{([^}]*)\}", user)
            if not got:
                return json.dumps({"episodic": [], "semantic": [], "working": ""})
            rnd = got[0][1]
            ans = {a: v for a, _, v in got}
            split = len(set(ans.values())) > 1
            return json.dumps({
                "episodic": [f"Round {rnd}: " + ", ".join(f"{a} answered {v}" for a, v in ans.items())],
                "semantic": [{"key": f"round{rnd}_answers", "value": ", ".join(f"{a}={v}" for a, v in ans.items()),
                              "confidence": "high", "scope": "episode"}],
                "working": ("Answers disagree; check the minority answer step by step." if split
                            else f"All agents agree on {next(iter(ans.values()))}."),
            })
        if system == prompts.EVOLVE_SYSTEM:
            me = re.search(r"Agent: (agent_[abc])", user).group(1)
            res = json.loads(re.search(r"Task-native result r_e:\n(\{.*?\})\n\nStructured Memory", user, re.S).group(1))
            c = {"round1_correct": res["my_first_answer_correct"]} if me == "agent_c" else \
                {"round1_correct": res["partners"]["agent_c"]["first_answer_correct"]}
            if me == "agent_c":
                strategies = (["Write out each step and check the arithmetic before giving a first answer."]
                              if not c["round1_correct"] else
                              ["Keep writing out each step: it made my first answer correct."])
                return json.dumps({"strategies": strategies, "partners": {
                    "agent_a": {"tendency": "shows full working; final answer matched the key", "reliability": "high"},
                    "agent_b": {"tendency": "shows full working; final answer matched the key", "reliability": "high"}}})
            tend = ("first answer was wrong, then corrected after reading worked solutions" if not c["round1_correct"]
                    else "first answer was correct and came with worked steps")
            return json.dumps({"strategies": ["Before agreeing, re-check any first-round answer that differs from the others."],
                               "partners": {"agent_c": {"tendency": tend,
                                                        "reliability": "low - unchecked first answers" if not c["round1_correct"] else "medium - improving"}}})
        # debaters
        g = find_gold(user)
        if "These are the solutions to the problem from other agents" in user:
            return f"Having checked the other solutions step by step, the answer is \\boxed{{{g}}}."
        if AGENTS["agent_c"] in user.split("\n\n")[0]:
            if "Write out each step" in user or "Keep writing out each step" in user:
                return f"Step 1: read the quantities. Step 2: compute carefully. So the answer is \\boxed{{{g}}}."
            wrong = str(int(float(g)) + 2) if g.lstrip("-").isdigit() else g + "0"
            return f"Quick estimate: \\boxed{{{wrong}}}"
        return f"Working through each quantity in turn, the answer is \\boxed{{{g}}}."

    return respond


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--out", default="runs/debate")
    ap.add_argument("--backend", choices=["anthropic", "openai", "mock"], default="mock")
    ap.add_argument("--model", default=None)
    ap.add_argument("--baseline", action="store_true", help="plug-in attached but online and evolve off")
    args = ap.parse_args()

    problems = load_problems()
    if args.backend == "anthropic":
        llm = AnthropicLLM(model=args.model or "claude-opus-5-5", effort="low")
    elif args.backend == "openai":
        llm = OpenAICompatLLM(model=args.model or "gpt-4.1-mini",
                              api_key=os.environ.get("OPENAI_API_KEY") or os.environ.get("OPEN_AI_KEY"),
                              base_url=os.environ.get("OPENAI_BASE_URL"))
    else:
        llm = ScriptedLLM(mock_respond_factory(problems))
    out = Path(args.out)
    plugin = EvolveMAS(out / "memory", llm, online=not args.baseline, evolve=not args.baseline)
    for a in AGENTS:
        plugin.register(a, profile=AGENTS[a])
    for ep in range(args.episodes):
        r = run_episode(plugin, llm, problems[ep % len(problems)])
        print(f"episode {ep}: {r['problem']} gold={r['gold']} team={r['team_answer']} "
              f"first-round correct {r['first_round_correct']}/3", flush=True)
    print("usage:", llm.usage)


if __name__ == "__main__":
    main()
