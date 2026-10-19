"""HumanEval, cooperative setting: planner / developer / tester (ChatDev-style).

Per problem: the planner writes a short plan from the prompt; the developer writes the
function from the prompt and the plan; the host runs HumanEval's check() in a subprocess;
the tester reads the test output and either approves or writes a fix note; the developer
revises at most twice. Every agent runs with the EvolveMAS plug-in.

    python bench/humaneval.py --dry-run --out /tmp/humaneval.json   # scripted models, no spend
    python bench/humaneval.py --episodes 3 --out docs/runs/humaneval.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]

from evolvemas import OpenAICompatLLM, ScriptedLLM, prompts  # noqa: E402
from recorder import Recorder, TracedEvolveMAS  # noqa: E402

TASK_IDS = ["HumanEval/10", "HumanEval/17", "HumanEval/19"]   # chosen before running
AGENT_MODEL, MEMORY_MODEL = "gpt-4.1-mini", "gpt-4o-mini"
MAX_REPAIRS = 2
AGENTS = {
    "planner": "Planner: reads the specification and writes a short implementation plan with edge cases.",
    "developer": "Developer: writes the Python function that implements the plan.",
    "tester": "Tester: reads the test results and either approves the code or explains what to fix.",
}
SYSTEM = "You are one member of a small software team that writes Python functions."


def load_problems():
    rows = [json.loads(l) for l in (HERE / "data" / "humaneval_problems.jsonl").read_text().splitlines() if l.strip()]
    by_id = {r["task_id"]: r for r in rows}
    return [by_id[t] for t in TASK_IDS]


def extract_code(text, prob):
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", text, re.S)
    code = max(blocks, key=len) if blocks else text
    if f"def {prob['entry_point']}" not in code:
        code = prob["prompt"] + code          # body only: put it under the given signature
    return code.strip() + "\n"


def run_tests(code, prob):
    program = code + "\n\n" + prob["test"] + f"\n\ncheck({prob['entry_point']})\nprint('TEST_PASSED')\n"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(program)
        path = f.name
    try:
        p = subprocess.run([sys.executable, path], capture_output=True, text=True, timeout=10)
        out = (p.stdout + p.stderr).strip()
        passed = p.returncode == 0 and "TEST_PASSED" in p.stdout
    except subprocess.TimeoutExpired:
        out, passed = "Timed out after 10 s.", False
    finally:
        os.unlink(path)
    # Keep the tail of a traceback; the test file path is a temp path and says nothing.
    out = re.sub(r'File "[^"]+\.py"', 'File "solution.py"', out)
    return passed, out[-1500:] or "(no output)"


# ------------------------------------------------------------------ dry-run backend
def scripted(system, user):
    if system == prompts.ONLINE_SYSTEM:
        return json.dumps({"episodic": ["Tests ran on the developer's code"],
                           "semantic": [{"key": "last_test", "value": "see evidence", "confidence": "low", "scope": "episode"}],
                           "working": "Watch the test output."})
    if system == prompts.EVOLVE_SYSTEM:
        return json.dumps({"strategies": ["State edge cases (empty input) in the plan before coding."],
                           "partners": {"developer": {"tendency": "follows the plan closely", "reliability": "high - one episode"}}})
    if "Planner" in user:
        return "Plan: handle the empty string, then build the result step by step."
    if "Tester" in user:
        return "APPROVE" if "TEST_PASSED" in user else "FIX: handle the empty input case."
    return "```python\ndef placeholder():\n    pass\n```"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--out", default="docs/runs/humaneval.json")
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

    problems = load_problems()[: args.episodes]
    rec = Recorder("humaneval", "HumanEval · planner, developer, tester",
                   source={"dataset": "HumanEval (openai/human-eval)", "task_ids": TASK_IDS[: args.episodes],
                           "protocol": f"plan -> code -> run check() -> tester review, up to {MAX_REPAIRS} repairs"},
                   models={"agents": "scripted" if args.dry_run else AGENT_MODEL,
                           "memory_agent": "scripted" if args.dry_run else MEMORY_MODEL})
    plugin = TracedEvolveMAS(Path(tempfile.mkdtemp(prefix="he_mem_")), mem_llm, rec)
    for aid, prof in AGENTS.items():
        plugin.register(aid, profile=prof)
        rec.agent(aid, label=aid.title(), role=prof)

    def ask(aid, body):
        mem = plugin.context(aid)
        user = f"Your role: {AGENTS[aid]}\n\n{body}" + (f"\n\n[Your memory]\n{mem}" if mem else "")
        reply = agent_llm(SYSTEM, user)
        rec.act(aid, user, reply)
        return reply

    for ep, prob in enumerate(problems):
        rec.episode({"task_id": prob["task_id"], "entry_point": prob["entry_point"], "prompt": prob["prompt"]})
        plugin.begin_episode()
        spec = f"Specification:\n```python\n{prob['prompt']}```"

        plan = ask("planner", f"{spec}\n\nWrite a short numbered plan (at most 6 steps) for implementing "
                               f"`{prob['entry_point']}`, including edge cases. Do not write code.")
        rec.msg("planner", ["developer"], plan)
        plugin.observe("planner", [{"speaker": "planner", "content": plan}])

        code_msg = ask("developer", f"{spec}\n\nPlan from the planner:\n{plan}\n\nWrite the complete function "
                                    "in one ```python block. Include needed imports and any helper functions.")
        rec.msg("developer", ["tester"], code_msg)
        plugin.observe("developer", [{"speaker": "planner", "content": plan}, {"speaker": "developer", "content": code_msg}])

        repairs, passed, history = 0, False, []
        while True:
            code = extract_code(code_msg, prob)
            passed, out = run_tests(code, prob)
            history.append({"attempt": repairs, "passed": passed})
            rec.env(f"Ran HumanEval check() on attempt {repairs + 1}: {'passed' if passed else 'failed'}.",
                    attempt=repairs + 1, passed=passed, output=out, code=code)
            review = ask("tester", f"{spec}\n\nDeveloper's code:\n```python\n{code}```\n\nTest output:\n{out}\n\n"
                                   "If the tests passed, reply with APPROVE and one sentence. Otherwise reply starting "
                                   "with FIX: and explain concisely what is wrong and how to fix it.")
            rec.msg("tester", ["developer"], review)
            plugin.observe("tester", [{"speaker": "developer", "content": code_msg},
                                      {"speaker": "env", "content": f"Test output: {out}"},
                                      {"speaker": "tester", "content": review}])
            if passed or repairs >= MAX_REPAIRS:
                plugin.observe("developer", [{"speaker": "tester", "content": review}])
                break
            repairs += 1
            code_msg = ask("developer", f"{spec}\n\nYour previous code:\n```python\n{code}```\n\nTester's note:\n{review}\n\n"
                                        "Rewrite the complete function in one ```python block.")
            rec.msg("developer", ["tester"], code_msg)
            plugin.observe("developer", [{"speaker": "tester", "content": review}, {"speaker": "developer", "content": code_msg}])
        plugin.observe("planner", [{"speaker": "env", "content": f"Final result: {'passed' if passed else 'failed'} after {repairs} repair(s)."}])

        result = {"passed": passed, "repairs": repairs, "attempts": history}
        rec.result(result)
        for aid in AGENTS:
            plugin.end_episode(aid, {"task_id": prob["task_id"], "passed": passed, "repairs_used": repairs})
        tokens = sum(l.usage["input_tokens"] + l.usage["output_tokens"] for l in llms)
        print(f"episode {ep}: {prob['task_id']} passed={passed} repairs={repairs}; tokens so far {tokens}", flush=True)
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
