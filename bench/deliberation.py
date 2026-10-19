"""LLM-Deliberation host for the EvolveMAS demo: the six-party "Harbour Sport Park" negotiation
(Abdelnabi et al., 2024), cooperative incentives, as used in the paper.

The protocol follows the original code (main.py, rounds.py, initial_prompts.py, prompt_utils.py,
save_utils.py): SportCo (p1) opens with the initial deal, parties then speak in a shuffled order
with a scratchpad that stays private and a short public answer, and SportCo makes the final
proposal. Scoring follows the paper's evaluation: a party agrees with SportCo's last complete
proposal when that deal's score is strictly above its minimum threshold.

All six parties are plugged into EvolveMAS. Before each of its turns, a party's MemoryAgent
observes the public answers it heard since its last turn; the memory and knowledge block is
appended to its game instructions.

    python bench/deliberation.py --dry-run --out /tmp/delib_dry.json
    python bench/deliberation.py --episodes 3 --out docs/runs/deliberation.json
"""
from __future__ import annotations

import argparse
import random
import re
import string
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mixed_common import AGENT_MODEL, MEMORY_MODEL, check_and_save, make_llms, usage_summary  # noqa: E402
from recorder import Recorder, TracedEvolveMAS  # noqa: E402

GAME = Path(__file__).resolve().parent / "data/deliberation"
ROUNDS = 16        # run_sessions.py default used for the paper's sessions: 1 opening + 16 + 1 final = 18 turns
WINDOW = 3         # run_sessions.py default history window
N_ISSUES = 5
N_AGENTS = 6


# ------------------------------------------------------------------ game setup (initial_prompts.py)
def load_game():
    agents = {}
    for line in (GAME / "config.txt").read_text().splitlines():
        if not line.strip():
            continue
        name, file_name, role, incentive = [x.strip() for x in line.split(",")[:4]]
        lines = (GAME / "scores_files" / f"{file_name}.txt").read_text().strip().split("\n")
        scores = {string.ascii_uppercase[i]: [int(v) for v in l.split(",")] for i, l in enumerate(lines[:-1])}
        agents[name] = {"file": file_name, "role": role, "incentive": incentive, "scores": scores, "min": int(lines[-1].strip())}
    p1 = next(n for n, a in agents.items() if a["role"] == "p1")
    p2 = next(n for n, a in agents.items() if a["role"] == "p2")
    initial_deal = (GAME / "initial_deal.txt").read_text().strip().splitlines()[0].strip()
    return agents, p1, p2, initial_deal


def initial_prompt(name, a, p1, p2):
    g = (GAME / "global_instructions.txt").read_text().replace(f'"{name}"', f'"{name}" (represented by you)')
    ind = (GAME / "individual_instructions" / a["incentive"] / f"{a['file']}.txt").read_text()
    for key, opts in a["scores"].items():
        for i, v in enumerate(opts):
            ind = ind.replace(f"#{key}{i + 1}_NUM", str(v))
        ind = ind.replace(f"#{key}_MAX_NUM", str(max(opts)))
    scoring = f"""
        Scoring rules:
        - The full deal has to involve one option per each issue.
        - You cannot accept any deal with a score less than {a['min']}. This is the minimum score you can accept.
        - You cannot under any circumstances disclose numbers in your scoring sheet or the values of the deal to the other parties. But you can share high-level priorities (e.g., you can say: I prefer this option, etc.).
        """
    voting = """
        Voting rules:
        - You interact with the other parties by taking turns to speak.
        - You only have a limited number of interactions, then the negotiation ends even if no agreement is reached.
        """
    voting += ("- Finally, you will consolidate all suggestions and pass a formal proposal for a test vote. " if name == p1
               else f"- Finally, {p1} will consolidate all suggestions and pass a formal proposal for a test vote. ")
    inc = """
        - The max score you can achieve is 100. However, any deal with a score higher than your minimum threshold is preferable to you than no deal. You are very open to any compromise to achieve that
        """
    if name == p1:
        inc += f'- Ensuring "{p2}"\'s approval is crucial because they have veto power. Focus on keys issues that appeal to them. '
        inc += f"\n\t- The proposal will pass if at least {N_AGENTS - 1} parties agree (must include {p2}). Your score will be this final deal's score. "
    elif name == p2:
        inc += f'- Ensuring "{p1}"\'s approval is crucial because they have veto power. Focus on keys issues that appeal to them. '
        inc += f"\n\t- The proposal will pass if at least {N_AGENTS - 1} parties agree (must include {p1}). Your score will be this final deal's score. "
    else:
        inc += f'- Ensuring "{p1}"\'s and "{p2}"\'s approval is crucial because they have veto power. Focus on keys issues that appeal to them. '
        inc += f"\n\t- The proposal will pass if at least {N_AGENTS - 1} parties agree (must include {p1} and the {p2}). Your score will be this final deal's score. "
    out = g + "\n" + ind + scoring + voting + inc
    if name == p1:
        out += f"\n\t- To protect yourself from potential future lawsuits, you want to achieve unanimity; if you and all other {N_AGENTS - 1} parties agree, you will get a bonus of 10 points. "
    return out


# ------------------------------------------------------------------ round prompts (rounds.py, prompt_utils.py)
COOP_SCRATCH = """
        Please use a scratchpad to show intermediate calculations and explain yourself and why you are agreeing with a deal or suggesting a new one.
        You should map the individual options to their scores denoted by the number between parentheses.
        You have a calculator tool at your disposal, where you simply add scores of the options to determine the total score of a deal.
        In your scratchpad,
            1) think about what others may prefer,
            2) Based on others' preferences and history and your notes, propose one proposal that balances between your scores and accommodating others and that is more likely to lead to an agreement.

        You must follow these important negotiation guidelines in all your suggestions:
        Aim for a balanced agreement considering all parties' interests.
        Show flexibility and openness to accommodate others' preferences.
        Express your objectives clearly and actively listen to others.
        Empathize with other parties' concerns to foster rapport.
        Focus on common interests to create a win-win situation.
        It is very important for you that you all reach an agreement as long as your minimum score is met."""

UNIFIED = '''
        Enclose the scratchpad between <SCRATCHPAD> and </SCRATCHPAD>. The scratchpad is secret and not seen by other parties.
        Your final answer is public and must never contain scores. Enclose your final answer after the scratchpad between <ANSWER> and </ANSWER>.
        Make your final answer very short and brief in 2-3 sentences and containing only your main proposals.
        Use options' short notations instead of long descriptions.
        Enclose any deals you suggest between: <DEAL> </DEAL>. "
        '''


def first_slot(deal, name):
    return (f" The negotiation now begins. As a representative of {name}, you are now talking to the other parties. "
            f"Use two to three short sentences overall. This is round: 0. To start, propose the following deal: {deal}. "
            f"Enclose the deal between: <DEAL> </DEAL> format. ")


def slot_prompt(name, p1, history, plans, round_idx, initial_deal):
    if round_idx == 0 and name == p1:
        return first_slot(initial_deal, name)
    final_round = (ROUNDS - round_idx) <= N_AGENTS
    final_vote = round_idx == ROUNDS
    parts = []
    for slot in history[-WINDOW:]:
        who = slot["agent"]
        parts.append(f". You ({who}): {slot['public_answer']}" if who == name else f". {who}: {slot['public_answer']}")
    joined = " \n ".join(parts)
    h = f"The following is a chronological history of up to {WINDOW} interactions <HISTORY> {joined} </HISTORY> "
    if plans.get(name):
        h += ("The following are your previous plans from last interactions. You should follow them while also adjusting "
              f"them according to new observations. <PREV_PLAN> {plans[name][-1]} </PREV_PLAN> ")
    h += "\n Now it is your turn to talk."
    if final_round:
        h += " This is the final discussion session."
    elif final_vote:
        h += " This is an official and final voting session."
    plan = ""
    if not final_round:
        plan = (" \n            After the final answer, building on your current move and analysis, \n            briefly write down short notes "
                "for yourself of what exact options you can explore the next time you speak. \n            Enclose the notes between <PLAN> and </PLAN>. ")
    elif name == p1 and not final_vote:
        plan = (" \n            After the final answer, building on your current move, \n            briefly write down short notes for yourself "
                "of what exact options you can suggest in the next and final voting session. \n            Enclose the notes between <PLAN> and </PLAN>. ")
    return h + COOP_SCRATCH + UNIFIED + plan


def extract_answer(full):
    a = full
    if "</ANSWER>" in a:
        a = a.split("<ANSWER>")[-1].split("</ANSWER>")[0]
    if "<ANSWER>" in a:
        a = a.split("<ANSWER>")[-1]
    return a.strip()


def extract_plan(full):
    if "<PLAN>" in full:
        return full.split("<PLAN>")[-1].split("</PLAN>")[0].strip()
    return ""


def turn_order(names, p1, seed):
    """randomize_agents_order from utils.py, seeded per session."""
    rng = random.Random(seed)
    out, last = [], p1
    for _ in range(-(-ROUNDS // len(names))):
        s = rng.sample(names, len(names))
        while s[0] == last or s[-1] == p1:
            s = rng.sample(names, len(names))
        out += s
        last = s[-1]
    return out


# ------------------------------------------------------------------ scoring (eval_paper_metrics.py)
def extract_deal(answer):
    answer = answer.replace("\n", "")
    deal = []
    for i in range(N_ISSUES):
        opts = re.findall(f"{string.ascii_uppercase[i]}[1-9]", answer)
        deal.append(opts[0] if opts else "")
    return deal, sum(1 for d in deal if d)


def calculator(scores, deal):
    total = 0
    for item in deal:
        if len(item) != 2 or item[0] not in scores or not 1 <= int(item[1]) <= len(scores[item[0]]):
            return 0
        total += scores[item[0]][int(item[1]) - 1]
    return total


def scripted(system, user):
    """Plumbing-only replies for --dry-run."""
    return "<SCRATCHPAD>dry run</SCRATCHPAD><ANSWER>I can support <DEAL>A2,B2,C3,D2,E3</DEAL>.</ANSWER><PLAN>try B2</PLAN>"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--out", default="docs/runs/deliberation.json")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    agent_llm, mem_llm, _ = make_llms(args.dry_run, scripted)
    agents, p1, p2, initial_deal = load_game()
    names = list(agents)

    rec = Recorder(
        "deliberation", "LLM-Deliberation · Harbour Sport Park",
        source={"dataset": "LLM-Deliberation (Abdelnabi et al., NeurIPS 2024), base game, cooperative incentives",
                "protocol": f"{ROUNDS} shuffled turns between SportCo's opening and final proposal, history window {WINDOW}",
                "note": "Game instructions and score sheets are the files used in the paper's runs."},
        models={"agents": AGENT_MODEL, "memory_agent": MEMORY_MODEL, "evolve": MEMORY_MODEL} if not args.dry_run
        else {"all": "scripted (dry run)"})
    for n, a in agents.items():
        rec.agent(n, n, {"p1": "p1 (proposer, veto)", "p2": "p2 (veto)"}.get(a["role"], "party"),
                  minimum=a["min"], scores=a["scores"], instructions=initial_prompt(n, a, p1, p2))

    store = Path(tempfile.mkdtemp(prefix="delib_mem_"))
    plugin = TracedEvolveMAS(store, mem_llm, rec, history=False)
    base = {}
    for n, a in agents.items():
        plugin.register(n, profile=f"{n}, a party in a six-party negotiation over a harbour sport park")
        base[n] = initial_prompt(n, a, p1, p2)

    for ep in range(args.episodes):
        order = turn_order(names, p1, seed=1000 + ep)[:ROUNDS]
        rec.episode({"initial_deal": initial_deal, "turn_order": [p1] + order + [p1], "p1": p1, "p2": p2,
                     "rounds": ROUNDS, "window": WINDOW})
        plugin.begin_episode()
        history, plans = [], {}
        pending = {n: [] for n in names}

        def take_turn(name, round_idx):
            if pending[name]:
                plugin.observe(name, pending[name])
                pending[name] = []
            mem = plugin.context(name)
            system = base[name] + (f"\n\n[Your memory]\n{mem}\n[End of memory]" if mem else "")
            user = slot_prompt(name, p1, history, plans, round_idx, initial_deal)
            full = agent_llm(system, user)
            public = extract_answer(full)
            plan = extract_plan(full)
            if plan:
                plans.setdefault(name, []).append(plan)
            history.append({"agent": name, "public_answer": public})
            rec.act(name, ("[Game instructions for " + name + ": see agent card]" + (f"\n\n[Your memory]\n{mem}\n[End of memory]" if mem else "")) + "\n\n" + user, full)
            rec.msg(name, [n for n in names if n != name], public)
            for n in names:
                pending[n].append({"speaker": name, "content": public})
            return public

        take_turn(p1, 0)                                     # opening deal
        for r in range(ROUNDS):
            take_turn(order[r], r)
        final = take_turn(p1, ROUNDS)                        # final proposal / vote
        for n in names:
            if pending[n]:
                plugin.observe(n, pending[n])
                pending[n] = []

        last_deal = None
        for h in history:
            if h["agent"] == p1:
                d, k = extract_deal(h["public_answer"])
                if k == N_ISSUES:
                    last_deal = d
        per = {n: calculator(a["scores"], last_deal) for n, a in agents.items()} if last_deal else {}
        agree = {n: per[n] > agents[n]["min"] for n in per} if last_deal else {}
        n_agree = sum(agree.values()) if last_deal else 0
        passed = False
        if last_deal:
            if n_agree == N_AGENTS:
                passed = True
            elif n_agree == N_AGENTS - 1 and not agree[p1] and per[p1] + 10 >= agents[p1]["min"]:
                passed = True                                # P1 bonus rule
            elif n_agree == N_AGENTS - 1 and agree[p1] and agree[p2]:
                passed = True
        deal_str = ",".join(last_deal) if last_deal else None
        rec.env("Scored SportCo's last complete proposal against every party's private threshold.",
                final_deal=deal_str, party_scores=per, thresholds={n: a["min"] for n, a in agents.items()}, agree=agree)
        result = {"final_deal": deal_str, "agreeing_parties": n_agree, "of": N_AGENTS, "passed": passed,
                  "turns": len(history)}
        rec.result(result)
        for n in names:
            plugin.end_episode(n, {"final_deal": deal_str, "your_score": per.get(n), "your_minimum": agents[n]["min"],
                                   "you_would_agree": agree.get(n), "agreeing_parties": n_agree, "of": N_AGENTS,
                                   "passed": passed})
        print(f"session {ep}: final deal {deal_str}, {n_agree}/6 agree, passed={passed}", flush=True)

    usage = usage_summary({"agents": agent_llm, "memory_agent": mem_llm})
    check_and_save(rec, args.out, usage)
    print("usage:", usage)


if __name__ == "__main__":
    main()
